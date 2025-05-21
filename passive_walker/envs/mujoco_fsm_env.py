# passive_walker/envs/mujoco_fsm_env.py
# Passive Walker Environment using MuJoCo and Gym
# A bipedal walker environment that can be controlled either by a Finite State Machine (FSM)
# or by a neural network controller. The walker operates on an inclined plane.

import os
import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw
import jax.numpy as jnp
from passive_walker.constants import XML_PATH


#––– Simulation Defaults ––––––––––––––––––––––––––––––
SIM_END       = 30.0     # Maximum simulation time in seconds
CONTROL_HZ    = 60       # Controller frequency in Hz (simulation steps per second)
RAMP_SLOPE    = 0.2      # Incline angle of the ramp in radians (approximately 11.5 degrees)

#––– FSM Targets ––––––––––––––––––––––––––––––––––––
HIP_SWING_POS   = 0.5    # Target hip angle for forward swing (radians)
HIP_SWING_NEG   = -0.5   # Target hip angle for backward swing (radians)
KNEE_STANCE     = 0.0    # Target knee angle during stance phase (radians)
KNEE_RETRACT    = -0.25  # Target knee angle during retraction phase (radians)


# ---------- Helper:  convert quaternion to Euler angles ----------------------------------------------
def quat2euler(q):
    """
    MuJoCo quaternion [w, x, y, z] → roll-pitch-yaw (rad).
    Safe for both np and jnp arrays.
    """
    w, x, y, z = q
    xp = jnp if isinstance(q, jnp.ndarray) else np   # backend picker

    # roll (x-axis)
    t0 = 2 * (w * x + y * z)
    t1 = 1 - 2 * (x * x + y * y)
    roll = xp.arctan2(t0, t1)

    # pitch (y-axis)
    t2 = 2 * (w * y - z * x)
    t2 = xp.clip(t2, -1.0, 1.0)
    pitch = xp.arcsin(t2)

    # yaw (z-axis)
    t3 = 2 * (w * z + x * y)
    t4 = 1 - 2 * (y * y + z * z)
    yaw = xp.arctan2(t3, t4)

    return roll, pitch, yaw

class PassiveWalkerEnv(gym.Env):
    """
    Gym environment for a passive walker using MuJoCo.
    
    This environment implements a bipedal walker that can be controlled in two ways:
    1. Using a Finite State Machine (FSM) for both hip and knee joints
    2. Using a neural network controller for either hip, knees, or both
    
    Control Modes:
    - When use_nn_for_hip is False, the hip joint is controlled by FSM logic
    - When use_nn_for_hip is True, the hip actuator is set by external actions
    - When use_nn_for_knees is False, the knee joints are controlled by FSM logic
    - When use_nn_for_knees is True, the knee actuators are set by external actions
    
    The external_action (when provided) is expected to be a 3-element vector:
    [hip_command, left_knee_command, right_knee_command]
    
    The environment simulates walking on an inclined plane, with gravity adjusted
    to create the slope effect.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, xml_path=XML_PATH, simend=SIM_END, use_nn_for_hip=False, use_nn_for_knees=False, use_gui=True):
        """
        Initialize the passive walker environment.
        
        Args:
            xml_path (str): Path to the MuJoCo XML model file
            simend (float): Maximum simulation time in seconds
            use_nn_for_hip (bool): Whether to use neural network control for hip
            use_nn_for_knees (bool): Whether to use neural network control for knees
            use_gui (bool): Whether to create a visualization window
        """
        super().__init__()
        self.simend = simend
        self.use_nn_for_hip = use_nn_for_hip
        self.use_nn_for_knees = use_nn_for_knees
        self.use_gui = use_gui

        # Initialize GLFW and create a window only if use_gui is True
        if self.use_gui:
            if not glfw.init():
                raise Exception("GLFW initialization failed")
            self.window = glfw.create_window(1200, 900, "Passive Walker Viewer", None, None)
            if not self.window:
                glfw.terminate()
                raise Exception("GLFW window creation failed")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
        else:
            self.window = None

        # Load the MuJoCo model and simulation data
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Define observation space: concatenation of qpos and qvel
        n_obs = self.model.nq + self.model.nv # number of observation 7+7=14
        #print("nq", self.model.nq)
        #print("nv", self.model.nv)
        #print(f"Observation space: {n_obs}")
        self.observation_space = spaces.Box(low=-np.inf, 
                                          high=np.inf, 
                                          shape=(n_obs,), 
                                          dtype=np.float32)
        
        # Setup rendering components if use_gui is True
        if self.use_gui:
            self.cam = mujoco.MjvCamera()
            self.cam.distance = 8.0
            self.opt = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        else:
            self.cam = None
            self.opt = None
            self.scene = None
            self.context = None
        
        # Retrieve MuJoCo model IDs for bodies and actuators
        self.hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        self.left_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_leg")
        self.right_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
        
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

        self.hip_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act")
        self.left_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.right_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        
        # FSM state definitions for hip control
        self.FSM_HIP_LEG1_SWING = 0  # Left leg swinging
        self.FSM_HIP_LEG2_SWING = 1  # Right leg swinging
        self.fsm_hip = self.FSM_HIP_LEG2_SWING  # Initial state
        
        # FSM state definitions for knee control
        self.FSM_KNEE1_STANCE = 0    # Left knee in stance phase
        self.FSM_KNEE1_RETRACT = 1   # Left knee in retraction phase
        self.FSM_KNEE2_STANCE = 0    # Right knee in stance phase
        self.FSM_KNEE2_RETRACT = 1   # Right knee in retraction phase
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE
        
        # Set gravity to simulate a ramp
        self.model.opt.gravity[0] = 9.81 * np.sin(RAMP_SLOPE)  # x-component for slope
        self.model.opt.gravity[1] = 0.0 # y-component for slope
        self.model.opt.gravity[2] = -9.81 * np.cos(RAMP_SLOPE) # z-component for gravity
        
    def init_controller(self):
        """
        Initialize the controller state.
        
        Sets the right leg's position and initializes the hip actuator based on FSM state.
        This is called during environment reset to ensure proper initial conditions.
        
        Returns:
            float: The initial desired hip position
        """
        self.data.qpos[self.right_leg_body_id] = HIP_SWING_POS
        # Set initial hip command using FSM
        desired_hip = HIP_SWING_POS if self.fsm_hip == self.FSM_HIP_LEG2_SWING else HIP_SWING_NEG
        self.data.ctrl[self.hip_pos_actuator_id] = desired_hip
        return desired_hip

    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Reset all the mjData arrays to model defaults
        mujoco.mj_resetData(self.model, self.data)
        # Apply initial controller settings
        self.demo_hip = self.init_controller()
        # Propagate qpos→xpos, xquat, etc
        mujoco.mj_forward(self.model, self.data)
        # Reset timer & FSM states
        self.data.time = 0.0
        self.fsm_hip   = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE
        return self._get_obs()
    
    def _get_obs(self):
        """
        Get the current observation of the environment.
        
        Returns:
            np.ndarray: Observation vector containing:
                - x position
                - z position of torso
                - torso pitch
                - x velocity
                - z velocity
                - hip angle
                - left knee angle
                - right knee angle
                - hip angular velocity
                - left knee angular velocity
                - right knee angular velocity
        """
        # Get torso position and orientation
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        x = float(self.data.qpos[0])  # x position
        z = float(self.data.xpos[torso_id, 2])  # z position
        quat = self.data.xquat[torso_id]  # [w,x,y,z]
        _, pitch, _ = quat2euler(quat)  # Extract pitch angle
        
        # Get velocities
        xdot = float(self.data.qvel[0])  # x velocity
        zdot = float(self.data.cvel[torso_id, 2])  # z velocity
        
        # Get joint angles
        hip_q = float(self.data.qpos[self.hip_pos_actuator_id])
        lk_q  = float(self.data.qpos[self.left_knee_pos_actuator_id])
        rk_q  = float(self.data.qpos[self.right_knee_pos_actuator_id])
        
        # Get joint velocities
        hip_qd = float(self.data.qvel[self.hip_pos_actuator_id])
        lk_qd  = float(self.data.qvel[self.left_knee_pos_actuator_id])
        rk_qd  = float(self.data.qvel[self.right_knee_pos_actuator_id])
        
        return np.array([
            x, z, pitch,
            xdot, zdot,
            hip_q, lk_q, rk_q,
            hip_qd, lk_qd, rk_qd
        ], dtype=np.float32)
    
    def controller_fsm_hip(self):
        """
        FSM logic for the hip joint control.
        
        Implements a state machine that switches between left and right leg swing phases
        based on foot contact and leg angles. The hip angle is set to either HIP_SWING_POS
        or HIP_SWING_NEG depending on the current state.
        """
        # Get left leg state
        quat_left = self.data.xquat[self.left_leg_body_id, :]
        euler_left = quat2euler(quat_left)
        abs_left = -euler_left[1]  # Negative because of coordinate system
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        # Get right leg state
        quat_right = self.data.xquat[self.right_leg_body_id, :]
        euler_right = quat2euler(quat_right)
        abs_right = -euler_right[1]  # Negative because of coordinate system
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        # State transitions based on foot contact and leg angles
        if self.fsm_hip == self.FSM_HIP_LEG2_SWING and pos_rightFoot[2] < 0.05 and abs_left < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG1_SWING
        elif self.fsm_hip == self.FSM_HIP_LEG1_SWING and pos_leftFoot[2] < 0.05 and abs_right < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG2_SWING

        # Set hip control based on current state
        if self.fsm_hip == self.FSM_HIP_LEG1_SWING:
            self.data.ctrl[self.hip_pos_actuator_id] = HIP_SWING_NEG
        else:
            self.data.ctrl[self.hip_pos_actuator_id] = HIP_SWING_POS

    def controller_fsm_knees(self):
        """
        FSM logic for the knee joints control.
        
        Implements a state machine for each knee that switches between stance and retraction
        phases based on foot contact and leg angles. The knee angles are set to either
        KNEE_STANCE or KNEE_RETRACT depending on the current state.
        """
        # Get left leg state
        quat_leftLeg = self.data.xquat[self.left_leg_body_id, :]
        euler_leftLeg = quat2euler(quat_leftLeg)
        abs_leftLeg = -euler_leftLeg[1]  # Negative because of coordinate system
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        # Get right leg state
        quat_rightLeg = self.data.xquat[self.right_leg_body_id, :]
        euler_rightLeg = quat2euler(quat_rightLeg)
        abs_rightLeg = -euler_rightLeg[1]  # Negative because of coordinate system
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        # Left knee state transitions
        if self.fsm_knee1 == self.FSM_KNEE1_STANCE and pos_rightFoot[2] < 0.05 and abs_leftLeg < 0.0:
            self.fsm_knee1 = self.FSM_KNEE1_RETRACT
        elif self.fsm_knee1 == self.FSM_KNEE1_RETRACT and abs_leftLeg > 0.1:
            self.fsm_knee1 = self.FSM_KNEE1_STANCE

        # Right knee state transitions
        if self.fsm_knee2 == self.FSM_KNEE2_STANCE and pos_leftFoot[2] < 0.05 and abs_rightLeg < 0.0:
            self.fsm_knee2 = self.FSM_KNEE2_RETRACT
        elif self.fsm_knee2 == self.FSM_KNEE2_RETRACT and abs_rightLeg > 0.1:
            self.fsm_knee2 = self.FSM_KNEE2_STANCE

        # Set knee controls based on current states
        if self.fsm_knee1 == self.FSM_KNEE1_STANCE:
            self.data.ctrl[self.left_knee_pos_actuator_id] = KNEE_STANCE
        else:
            self.data.ctrl[self.left_knee_pos_actuator_id] = KNEE_RETRACT

        if self.fsm_knee2 == self.FSM_KNEE2_STANCE:
            self.data.ctrl[self.right_knee_pos_actuator_id] = KNEE_STANCE
        else:
            self.data.ctrl[self.right_knee_pos_actuator_id] = KNEE_RETRACT

    def step(self, external_action):
        """
        Advance the simulation by one time step.
        
        Args:
            external_action (np.ndarray): Control signals for the joints:
                - If use_nn_for_hip is True, first element controls hip
                - If use_nn_for_knees is True, second and third elements control left and right knees
                - If a joint is not controlled by NN, FSM controller is used instead
        
        Returns:
            tuple: (observation, reward, done, info)
                - observation: Current state observation
                - reward: Reward for this step (forward progress)
                - done: Whether the episode is finished
                - info: Additional information (time)
        """
        # Calculate number of simulation steps needed for this control step
        sim_steps = int((1.0 / CONTROL_HZ) / self.model.opt.timestep)
        
        # Simulate for the required number of steps
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
            # Handle hip control
            if self.use_nn_for_hip:
                self.data.ctrl[self.hip_pos_actuator_id] = external_action[0]
            else:
                self.controller_fsm_hip()
                
            # Handle knee control
            if self.use_nn_for_knees:
                # If hip is also NN-controlled, knees are at indices 1,2
                # If only knees are NN-controlled, they are at indices 0,1
                knee_start_idx = 1 if self.use_nn_for_hip else 0
                self.data.ctrl[self.left_knee_pos_actuator_id] = external_action[knee_start_idx]
                self.data.ctrl[self.right_knee_pos_actuator_id] = external_action[knee_start_idx + 1]
            else:
                self.controller_fsm_knees()

        obs = self._get_obs()
        reward = self.data.qpos[0]  # Reward is forward progress
        done = self.data.time >= self.simend
        info = {"time": self.data.time}
        return obs, reward, done, info

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        
        Args:
            mode (str): Rendering mode ("human" or "rgb_array")
        """
        if not self.use_gui:
            return
            
        # Update viewport and camera
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        self.cam.lookat[0] = self.data.qpos[0]  # Follow the walker
        
        # Update scene and render
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                              mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        # Update display
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """
        Clean up resources and close the environment.
        """
        if self.use_gui and self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()

if __name__ == "__main__":
    # Testing the environment
    import os
    dirname = os.path.dirname(__file__)
    
    # Test demo mode (FSM for both hip and knees) with GUI
    print("Testing FSM mode (demo) with GUI:")
    env_demo = PassiveWalkerEnv(str(XML_PATH), simend=10, use_nn_for_hip=False, use_nn_for_knees=False, use_gui=True)
    obs = env_demo.reset()
    done = False
    total_reward = 0.0
    
    # Run simulation until done or window is closed
    while not done and not glfw.window_should_close(env_demo.window):
        obs, reward, done, info = env_demo.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        total_reward += reward
        print(f"Time: {info['time']:.3f} | Reward: {reward:.3f} | Total: {total_reward:.3f}")
        env_demo.render(mode="human")
    
    env_demo.close()
    print("Demo mode with GUI finished.\n")
