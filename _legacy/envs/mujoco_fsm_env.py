"""
VLL Passive Walker Environment with FSM Control

A bipedal walker environment that supports both FSM and neural network control.
Features 11D observation space and PD control with neutral general actuators.
"""

import os
import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw
import jax.numpy as jnp

from passive_walker.constants import XML_PATH
from passive_walker.utils.control import (
    denormalize_action,
    compute_pd_control,
    get_joint_ranges,
    get_pd_gains,
    get_ctrl_limits,
)
from passive_walker.utils.jax_control import compute_pd_control_jax, compute_reward_jax, quat2euler_zyx_jit

# ============================================================================
# Constants
# ============================================================================

# Simulation parameters
SIM_END = 30.0      # Maximum simulation time (seconds)
CONTROL_HZ = 60    # Control frequency (Hz)

# FSM thresholds (hoisted to avoid repeated allocations)
CONTACT_HEIGHT = 0.05  # Contact detection height threshold
KNEE_RELEASE_THRESHOLD = 0.1  # Knee release threshold
RAMP_SLOPE = float(np.deg2rad(10.5))    # Ramp angle (radians) ~11.5°

# FSM target positions
HIP_SWING_POS = 0.5    # Hip forward swing (radians)
HIP_SWING_NEG = -0.5   # Hip backward swing (radians)
KNEE_STANCE = 0.0      # Knee stance phase (meters)
KNEE_RETRACT = -0.25   # Knee retraction phase (meters)

# ============================================================================
# Utility Functions
# ============================================================================

def pd_step_np(q, qd, q_des, kp, kv, umin, umax):
    """
    NumPy PD control step (faster than JAX for small vectors).
    
    Args:
        q: Current joint positions (3,)
        qd: Current joint velocities (3,)
        q_des: Desired joint positions (3,)
        kp: Proportional gains (3,)
        kv: Derivative gains (3,)
        umin: Minimum control limits (3,)
        umax: Maximum control limits (3,)
        
    Returns:
        Control torques/forces (3,)
    """
    u = kp * (q_des - q) - kv * qd
    return np.clip(u, umin, umax)

def quat2euler(q):
    """
    Convert MuJoCo quaternion [w,x,y,z] to roll-pitch-yaw (radians) using optimized NumPy.
    
    For per-step FSM operations, NumPy is faster than JAX due to zero compilation overhead.
    JAX version is kept for batched operations only.
    
    Args:
        q: Quaternion array [w, x, y, z]
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Extract components
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Direct Euler angle formulas (no 3x3 matrix needed)
    # Roll (rotation around x-axis)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    
    # Pitch (rotation around y-axis)
    t2 = 2 * (w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)  # Keep within [-1, 1] for arcsin
    pitch = np.arcsin(t2)
    
    # Yaw (rotation around z-axis)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    return roll, pitch, yaw


# ============================================================================
# Environment Class
# ============================================================================

class PassiveWalkerEnv(gym.Env):
    """
    VLL Passive Walker Environment with FSM Control.
    
    This environment implements a bipedal walker that can be controlled using:
    - Finite State Machine (FSM) for both hip and knee joints
    - Neural network controller for either hip, knees, or both
    
    Control Modes:
        FSM: Uses finite state machine for hip/knee control
        NN:  Uses neural network actions for hip/knee control
    
    Action API: 3D [hip, left_knee, right_knee] in [-1, 1] mapping to physical ranges via denorm
    Observation: 11D [x, z, pitch, ẋ, ż, hip_q, lk_q, rk_q, hip_q̇, lk_q̇, rk_q̇]
    """
    
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(
        self,
        xml_path=str(XML_PATH),
        simend=SIM_END,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=True,
    ):
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
        self.ctrl_hz = CONTROL_HZ

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
        
        # Define observation space to match _get_obs() output (11D)
        # Note: We use 11D observation even though model has 6 joints (12 states total)
        # because we exclude the left_leg_lock joint which was removed
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # Define action space for the 3 actuators (hip and two knees)
        # Actions are normalized [-1, 1] and denormalized to physical ranges via denormalize_action()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
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
        
        # Retrieve MuJoCo model IDs for joints and bodies
        self.hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        self.left_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.right_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        
        # Get slide joint IDs for velocity computation
        self.slide_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_x")
        self.slide_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slide_z")
        self.pitch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "pitch")
        
        # Cache joint addresses for qpos/qvel indexing
        self.hip_qpos_addr = self.model.jnt_qposadr[self.hip_id]
        self.hip_qvel_addr = self.model.jnt_dofadr[self.hip_id]
        self.left_knee_qpos_addr = self.model.jnt_qposadr[self.left_knee_id]
        self.left_knee_qvel_addr = self.model.jnt_dofadr[self.left_knee_id]
        self.right_knee_qpos_addr = self.model.jnt_qposadr[self.right_knee_id]
        self.right_knee_qvel_addr = self.model.jnt_dofadr[self.right_knee_id]
        
        # Cache slide joint addresses for velocity computation
        self.slide_x_dof = self.model.jnt_dofadr[self.slide_x_id]
        self.slide_z_dof = self.model.jnt_dofadr[self.slide_z_id]
        self.pitch_qpos_addr = self.model.jnt_qposadr[self.pitch_id]
        self.pitch_qvel_addr = self.model.jnt_dofadr[self.pitch_id]
        
        self.left_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_leg")
        self.right_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
        
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        
        # Pre-allocate observation buffer for zero-allocation observations
        self._obs = np.empty(11, dtype=np.float32)
        
        # Pre-allocate scratch arrays (zero per-step allocations)
        self._q = np.empty(3, dtype=np.float32)      # Current joint positions
        self._qd = np.empty(3, dtype=np.float32)     # Current joint velocities
        self._qdes = np.empty(3, dtype=np.float32)   # Desired joint positions
        self._u = np.empty(3, dtype=np.float32)      # Control torques/forces
        self._last_controls = np.empty(3, dtype=np.float32)  # Last applied controls
        
        # Precompute affine denormalization maps (micro-perf optimization)
        joint_ranges = get_joint_ranges()
        self._denorm_a = np.array([(hi-lo)/2 for (lo,hi) in joint_ranges], dtype=np.float32)
        self._denorm_b = np.array([(hi+lo)/2 for (lo,hi) in joint_ranges], dtype=np.float32)
        
        # Cache PD gains and limits as NumPy arrays (faster than JAX for small vectors)
        self._kp = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)  # hip, left_knee, right_knee
        self._kv = np.array([1.0, 100.0, 100.0], dtype=np.float32)
        self._umin = np.array([-50.0, -800.0, -800.0], dtype=np.float32)
        self._umax = np.array([50.0, 800.0, 800.0], dtype=np.float32)

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
        


    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            np.ndarray: Initial observation
        """
        # Reset all the mjData arrays to model defaults
        mujoco.mj_resetData(self.model, self.data)
        
        # Reset FSM states first
        self.fsm_hip   = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE
        
        # Set initial joint states using joint addresses
        self.data.qpos[self.hip_qpos_addr] = HIP_SWING_NEG
        self.data.qpos[self.left_knee_qpos_addr] = 0.0
        self.data.qpos[self.right_knee_qpos_addr] = 0.0
        
        # Zero all control signals (torques/forces)
        self.data.ctrl[self.hip_pos_actuator_id] = 0.0
        self.data.ctrl[self.left_knee_pos_actuator_id] = 0.0
        self.data.ctrl[self.right_knee_pos_actuator_id] = 0.0
        
        # Propagate qpos→xpos, xquat, etc
        mujoco.mj_forward(self.model, self.data)
        
        # Reset timer
        self.data.time = 0.0
        
        # Initialize previous x position for delta reward calculation
        self.prev_x = self.data.qpos[self.slide_x_dof]
        
        return self._get_obs()
    
    def _get_obs(self):
        """
        Get the current observation of the environment (allocation-free, branch-free).
        
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
        # Use pre-allocated buffer for zero-allocation observations
        ob = self._obs
        
        # Direct array indexing - no temporary variables, no allocations
        ob[0]  = self.data.qpos[self.slide_x_dof]      # x position
        ob[1]  = self.data.qpos[self.slide_z_dof]      # z position
        ob[2]  = self.data.qpos[self.pitch_qpos_addr]  # pitch angle
        ob[3]  = self.data.qvel[self.slide_x_dof]      # x velocity
        ob[4]  = self.data.qvel[self.slide_z_dof]      # z velocity
        ob[5]  = self.data.qpos[self.hip_qpos_addr]    # hip angle
        ob[6]  = self.data.qpos[self.left_knee_qpos_addr]  # left knee angle
        ob[7]  = self.data.qpos[self.right_knee_qpos_addr] # right knee angle
        ob[8]  = self.data.qvel[self.hip_qvel_addr]    # hip angular velocity
        ob[9]  = self.data.qvel[self.left_knee_qvel_addr]  # left knee angular velocity
        ob[10] = self.data.qvel[self.right_knee_qvel_addr] # right knee angular velocity
        
        return ob


    def _update_fsm_hip_state(self):
        """
        Update FSM hip state without setting control values.
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
        if self.fsm_hip == self.FSM_HIP_LEG2_SWING and pos_rightFoot[2] < CONTACT_HEIGHT and abs_left < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG1_SWING
        elif self.fsm_hip == self.FSM_HIP_LEG1_SWING and pos_leftFoot[2] < CONTACT_HEIGHT and abs_right < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG2_SWING

    def _update_fsm_knee_states(self):
        """
        Update FSM knee states without setting control values.
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
        if self.fsm_knee1 == self.FSM_KNEE1_STANCE and pos_rightFoot[2] < CONTACT_HEIGHT and abs_leftLeg < 0.0:
            self.fsm_knee1 = self.FSM_KNEE1_RETRACT
        elif self.fsm_knee1 == self.FSM_KNEE1_RETRACT and abs_leftLeg > KNEE_RELEASE_THRESHOLD:
            self.fsm_knee1 = self.FSM_KNEE1_STANCE

        # Right knee state transitions
        if self.fsm_knee2 == self.FSM_KNEE2_STANCE and pos_leftFoot[2] < CONTACT_HEIGHT and abs_rightLeg < 0.0:
            self.fsm_knee2 = self.FSM_KNEE2_RETRACT
        elif self.fsm_knee2 == self.FSM_KNEE2_RETRACT and abs_rightLeg > KNEE_RELEASE_THRESHOLD:
            self.fsm_knee2 = self.FSM_KNEE2_STANCE

    def _apply_pd_control(self, external_action):
        """
        Apply PD control based on action and FSM state.
        
        Args:
            external_action: 3-D normalized action vector [hip, left_knee, right_knee] in [-1, 1] 
                            that gets denormalized to physical ranges via denormalize_action()
        """
        # Get current joint states (no allocations)
        self._q[0] = self.data.qpos[self.hip_qpos_addr]
        self._q[1] = self.data.qpos[self.left_knee_qpos_addr]
        self._q[2] = self.data.qpos[self.right_knee_qpos_addr]
        
        self._qd[0] = self.data.qvel[self.hip_qvel_addr]
        self._qd[1] = self.data.qvel[self.left_knee_qvel_addr]
        self._qd[2] = self.data.qvel[self.right_knee_qvel_addr]
        
        # Determine desired positions (using scratch array)
        self._qdes.fill(0.0)
        
        # Hip control
        if self.use_nn_for_hip:
            # Denormalize action to physical range
            joint_ranges = get_joint_ranges()
            self._qdes[0] = denormalize_action([external_action[0]], [joint_ranges[0]])[0]
        else:
            # Use FSM logic to determine desired position (without setting control)
            self._update_fsm_hip_state()
            if self.fsm_hip == self.FSM_HIP_LEG1_SWING:
                self._qdes[0] = HIP_SWING_NEG  # FSM sets -0.5
            else:
                self._qdes[0] = HIP_SWING_POS   # FSM sets 0.5
        
        # Knee control
        if self.use_nn_for_knees:
            # Denormalize actions to physical ranges
            joint_ranges = get_joint_ranges()
            self._qdes[1] = denormalize_action([external_action[1]], [joint_ranges[1]])[0]
            self._qdes[2] = denormalize_action([external_action[2]], [joint_ranges[2]])[0]
        else:
            # Use FSM logic to determine desired positions (without setting control)
            self._update_fsm_knee_states()
            self._qdes[1] = KNEE_STANCE if self.fsm_knee1 == self.FSM_KNEE1_STANCE else KNEE_RETRACT
            self._qdes[2] = KNEE_STANCE if self.fsm_knee2 == self.FSM_KNEE2_STANCE else KNEE_RETRACT
        
        # Compute PD control using NumPy (faster for small vectors)
        self._u[:] = pd_step_np(self._q, self._qd, self._qdes, self._kp, self._kv, self._umin, self._umax)
        
        # Apply controls to actuators
        self.data.ctrl[self.hip_pos_actuator_id] = self._u[0]
        self.data.ctrl[self.left_knee_pos_actuator_id] = self._u[1]
        self.data.ctrl[self.right_knee_pos_actuator_id] = self._u[2]
        
        # Store controls for reward calculation
        self._last_controls[:] = self._u

    def step(self, external_action):
        """
        Advance the simulation by one time step.
        
        Args:
            external_action (np.ndarray): 3-D normalized control signals [hip, left_knee, right_knee] in [-1, 1]
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Calculate simulation steps
        sim_steps = max(1, int((1.0 / CONTROL_HZ) / self.model.opt.timestep))
        
        # Prebind hot attributes for performance
        data = self.data
        qpos = data.qpos
        qvel = data.qvel
        ctrl = data.ctrl
        
        # Prebind scratch arrays
        q = self._q
        qd = self._qd
        qdes = self._qdes
        u = self._u
        
        # Prebind joint addresses
        hip_qpos_addr = self.hip_qpos_addr
        left_knee_qpos_addr = self.left_knee_qpos_addr
        right_knee_qpos_addr = self.right_knee_qpos_addr
        hip_qvel_addr = self.hip_qvel_addr
        left_knee_qvel_addr = self.left_knee_qvel_addr
        right_knee_qvel_addr = self.right_knee_qvel_addr
        
        # Prebind actuator IDs
        hip_act_id = self.hip_pos_actuator_id
        left_knee_act_id = self.left_knee_pos_actuator_id
        right_knee_act_id = self.right_knee_pos_actuator_id
        
        # Determine control mode once (branch hoisting)
        use_nn_hip = self.use_nn_for_hip
        use_nn_knees = self.use_nn_for_knees
        
        # Precompute desired positions if using NN (vectorized affine denorm)
        if use_nn_hip or use_nn_knees:
            if use_nn_hip:
                qdes[0] = self._denorm_a[0] * external_action[0] + self._denorm_b[0]
            if use_nn_knees:
                qdes[1] = self._denorm_a[1] * external_action[1] + self._denorm_b[1]
                qdes[2] = self._denorm_a[2] * external_action[2] + self._denorm_b[2]
        
        # Optimized hot loop
        for _ in range(sim_steps):
            # Read current state (no allocations)
            q[0] = qpos[hip_qpos_addr]
            q[1] = qpos[left_knee_qpos_addr]
            q[2] = qpos[right_knee_qpos_addr]
            qd[0] = qvel[hip_qvel_addr]
            qd[1] = qvel[left_knee_qvel_addr]
            qd[2] = qvel[right_knee_qvel_addr]
            
            # Always update FSM states (needed for proper state transitions)
            if not use_nn_hip:
                self._update_fsm_hip_state()
            if not use_nn_knees:
                self._update_fsm_knee_states()
            
            # Fill desired positions
            if not use_nn_hip:
                # FSM hip control
                qdes[0] = HIP_SWING_NEG if self.fsm_hip == self.FSM_HIP_LEG1_SWING else HIP_SWING_POS
            
            if not use_nn_knees:
                # FSM knee control
                qdes[1] = KNEE_STANCE if self.fsm_knee1 == self.FSM_KNEE1_STANCE else KNEE_RETRACT
                qdes[2] = KNEE_STANCE if self.fsm_knee2 == self.FSM_KNEE2_STANCE else KNEE_RETRACT
            
            # NumPy PD control (faster for small vectors)
            u[:] = pd_step_np(q, qd, qdes, self._kp, self._kv, self._umin, self._umax)
            
            # Apply controls
            ctrl[hip_act_id] = u[0]
            ctrl[left_knee_act_id] = u[1]
            ctrl[right_knee_act_id] = u[2]
            
            # Advance physics
            mujoco.mj_step(self.model, data)

        # Store controls for reward calculation
        self._last_controls[:] = u
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward components (using prebound variables)
        current_x = qpos[self.slide_x_dof]
        dx = current_x - self.prev_x  # Forward progress
        
        # Get additional reward components
        pitch = float(qpos[self.pitch_qpos_addr])
        pitch_abs = abs(pitch)
        ctrl_abs_sum = np.sum(np.abs(u))  # Use current controls
        vx = float(qvel[self.slide_x_dof])  # Forward velocity
        
        # Get foot positions for clearance bonus
        left_foot_z = data.xpos[self.left_foot_body_id, 2]
        right_foot_z = data.xpos[self.right_foot_body_id, 2]
        torso_z = data.xpos[self.torso_body_id, 2]
        
        # FSM Environment: Minimal reward for data collection
        # Focus on data quality, not reward optimization
        
        # Simple forward progress reward (for logging purposes)
        reward = dx  # Just forward progress, no complex shaping
        
        # Log quality metrics for offline filtering (not used in reward)
        fell = (pitch_abs > 1.0) or (torso_z < 0.15)
        stalled = abs(vx) < 0.005
        unstable = pitch_abs > 0.5  # Mark as unstable for filtering
        
        # Episode termination
        done = fell or (data.time >= self.simend)
        # Optionally include stall: done = fell or stalled or (data.time >= self.simend)
        
        # Rich logging for data collection and offline filtering
        info = {
            'time': data.time,
            'total_reward': self.total_reward if hasattr(self, 'total_reward') else 0.0,
            'step_time': 0.0,  # Placeholder for step time
            'dx': dx,
            'pitch_abs': pitch_abs,
            'torso_z': torso_z,
            'left_foot_z': left_foot_z,
            'right_foot_z': right_foot_z,
            'vx': vx,
            'fell': fell,
            'stalled': stalled,
            'unstable': unstable,
            'fsm_hip_state': self.fsm_hip,
            'fsm_left_knee_state': getattr(self, 'fsm_left_knee', 0),
            'fsm_right_knee_state': getattr(self, 'fsm_right_knee', 0),
            'quality_score': self._compute_quality_score(pitch_abs, vx, left_foot_z, right_foot_z)
        }
        
        # Update reference for next step
        self.prev_x = current_x
        
        return obs, reward, done, info

    def _compute_quality_score(self, pitch_abs, vx, left_foot_z, right_foot_z):
        """
        Compute a quality score for data collection filtering.
        Higher scores indicate better quality data for imitation learning.
        """
        # Stability score (prefer low pitch)
        stability_score = max(0.0, 1.0 - pitch_abs / 0.5)  # 0-1, higher is better
        
        # Motion score (prefer reasonable velocity)
        motion_score = 1.0 if 0.1 <= abs(vx) <= 2.0 else 0.5  # Prefer walking speed
        
        # Foot clearance score (prefer alternating foot contacts)
        foot_clearance = min(left_foot_z, right_foot_z)
        clearance_score = 1.0 if foot_clearance > 0.02 else 0.0  # Both feet off ground
        
        # Combined quality score (0-3, higher is better)
        return stability_score + motion_score + clearance_score

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
    import time
    
    # Configuration for printing
    PRINT_ENABLED = True  # Set to False to disable all printing
    PRINT_INTERVAL = 2.0  # Print every 2 seconds (less frequent)
    
    # Test demo mode (FSM for both hip and knees) with GUI
    if PRINT_ENABLED:
        print("Testing FSM mode (demo) with GUI:")
    
    env_demo = PassiveWalkerEnv(str(XML_PATH), simend=50, use_nn_for_hip=False, use_nn_for_knees=False, use_gui=True)
    obs = env_demo.reset()
    done = False
    total_reward = 0.0
    
    # Throttle printing (configurable interval)
    t0 = time.time()
    last_print = t0
    n = 0
    
    # Run simulation until done or window is closed
    while not done and (not env_demo.use_gui or not glfw.window_should_close(env_demo.window)):
        step_start = time.time()
        obs, reward, done, info = env_demo.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        step_duration = time.time() - step_start
        total_reward += reward
        n += 1
        
        # Print at configurable intervals
        if PRINT_ENABLED:
            now = time.time()
            if now - last_print >= PRINT_INTERVAL:
                elapsed = now - t0
                fps = n / (now - last_print)
                print(f"Time: {info['time']:.3f} | Reward: {reward:.3f} | Total: {total_reward:.3f} | "
                      f"Step time: {step_duration:.4f}s | FPS: {fps:.2f}")
                last_print = now
                n = 0
        
        env_demo.render(mode="human")
    
    env_demo.close()
    if PRINT_ENABLED:
        print("Demo mode with GUI finished.")
        print(f"Total frames: {n}, Total time: {time.time() - t0:.2f}s, Average FPS: {n / (time.time() - t0):.2f}")
