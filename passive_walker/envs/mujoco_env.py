# passive_walker/envs/mujoco_env.py
# Passive Walker Environment using MuJoCo and Gym

import os
import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw
from scipy.spatial.transform import Rotation as R
import jax, jax.numpy as jnp
from brax.envs import base

# ---------- Constants --------------------------------------------------
UPRIGHT_PITCH_MAX = 0.20           # rad
FALL_Z_MIN        = 0.15           # m
FALL_PITCH_MAX    = 0.70           # ≈30°
ACT_COST_COEFF    = 1e-3
FOOT_CLEAR_THRESH = 0.03           # m
RAMP_DEG_MAX      = 15.0
FRICTION_MIN_MAX  = (0.6, 1.2)
MASS_JITTER       = 0.10           # ±10 %


# ---------- Helper:  fallen? ----------------------------------------------
def _is_fallen(ps):
    torso_z   = ps.x.pos[0, 2]
    # rotation matrix → pitch
    pitch     = jnp.arctan2(
                   2*ps.x.rot[0,0]*ps.x.rot[0,2] + 2*ps.x.rot[0,1]*ps.x.rot[0,3],
                   1 - 2*(ps.x.rot[0,2]**2 + ps.x.rot[0,1]**2)
               )
    return jnp.logical_or(torso_z < FALL_Z_MIN,
                          jnp.abs(pitch) > FALL_PITCH_MAX)

# ---------- Helper:  Helper: convert quaternion to Euler angles ----------------------------------------------

def quat2euler(quat):
    # MuJoCo: [w, x, y, z] --> SciPy expects [x, y, z, w]
    _quat = np.concatenate([quat[1:], quat[:1]])
    r = R.from_quat(_quat)
    return r.as_euler('xyz', degrees=False)


# ---------- Helper:  shaped reward ----------------------------------------
def _compute_reward(ps, act, metrics):
    x_now     = ps.x.pos[0, 0]
    dx        = x_now - metrics['prev_x']            # forward progress

    # upright bonus
    upright_bonus = jnp.where(metrics['pitch_abs'] < UPRIGHT_PITCH_MAX, 0.5, 0.0)

    # action penalty (act is 3‑D scaled action ∈[‑1,1])
    act_cost  = ACT_COST_COEFF * jnp.sum(jnp.abs(act))

    # foot‑drag bonus
    foot_z    = ps.x.pos[1:, 2]                      # both feet
    foot_clear= jnp.where(jnp.all(foot_z > FOOT_CLEAR_THRESH), 0.2, 0.0)

    return dx + upright_bonus + foot_clear - act_cost


# ---------- Helper:  apply domain randomisation in reset -------------------
def _randomise_physics(self, rng):
    rng, sub = jax.random.split(rng)
    tilt_deg = jax.random.uniform(sub, (), minval=0, maxval=RAMP_DEG_MAX)
    tilt_rad = jnp.deg2rad(tilt_deg)
    # rotate gravity vector around +y
    self.sys.gravity = jnp.array([9.81*jnp.sin(tilt_rad), 0, -9.81*jnp.cos(tilt_rad)])

    rng, sub = jax.random.split(rng)
    friction = jax.random.uniform(sub, (), minval=FRICTION_MIN_MAX[0],
                                          maxval=FRICTION_MIN_MAX[1])
    self.sys.geom_friction = jnp.full_like(self.sys.geom_friction, friction)

    rng, sub = jax.random.split(rng)
    mass_jit = jax.random.uniform(sub, (), minval=1-MASS_JITTER, maxval=1+MASS_JITTER)
    self.sys.link.inertia = self.sys.link.inertia.at[0].set(
        self.sys.link.inertia[0] * mass_jit)         # scale torso mass only
    return rng


class PassiveWalkerEnv(gym.Env):
    """
    Gym environment for a passive walker using MuJoCo.
    
    - When use_nn_for_hip is False, the hip joint is controlled by FSM logic.
      When True, the hip actuator is set externally.
    - When use_nn_for_knees is False, the knee joints are controlled by FSM logic.
      When True, the knee actuators are set externally.
    - The external_action (when provided) is expected to be a 3-element vector:
      [hip_command, left_knee_command, right_knee_command].
    - The use_gui flag controls whether a GLFW window is created for rendering.
    """
    metadata = {"render.modes": ["human", "rgb_array"]}
    
    def __init__(self, xml_path, simend=30.0, use_nn_for_hip=False, use_nn_for_knees=False, use_gui=True,
                 randomize_legs=False,hip_init_range=(0.2, 0.5),knee_init_range=(-0.25, 0.0)):
        
        super().__init__()
        self.simend = simend
        self.use_nn_for_hip = use_nn_for_hip
        self.use_nn_for_knees = use_nn_for_knees
        self.use_gui = use_gui

        # Initialize GLFW and create a window only if use_gui is True.
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

        # Load the MuJoCo model and simulation data.
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define observation space: concatenation of qpos and qvel.
        n_obs = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)
        # Dummy action space; required by Gym.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # Setup rendering components if use_gui is True.
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
        

        self.randomize_legs = randomize_legs
        self.hip_init_range = hip_init_range
        self.knee_init_range = knee_init_range
        
        # Retrieve IDs.
        self.hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        self.left_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_leg")
        self.right_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

        self.hip_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act")
        self.left_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.right_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        
        # FSM state definitions for hip control.
        self.FSM_HIP_LEG1_SWING = 0
        self.FSM_HIP_LEG2_SWING = 1
        self.fsm_hip = self.FSM_HIP_LEG2_SWING  # Initial state.
        
        # FSM state definitions for knee control.
        self.FSM_KNEE1_STANCE = 0
        self.FSM_KNEE1_RETRACT = 1
        self.FSM_KNEE2_STANCE = 0
        self.FSM_KNEE2_RETRACT = 1
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE
        
        # Set gravity to simulate a ramp.
        self.model.opt.gravity[0] = 9.81 * np.sin(0.2)
        self.model.opt.gravity[2] = -9.81 * np.cos(0.2)

        
    def init_controller(self):
        """
        Initialize the controller state.
        Set the right leg's position and initialize the hip actuator based on FSM.
        """
        self.data.qpos[self.right_leg_body_id] = 0.5
        # Set initial hip command using FSM.
        desired_hip = 0.5 if self.fsm_hip == self.FSM_HIP_LEG2_SWING else -0.5
        self.data.ctrl[self.hip_pos_actuator_id] = desired_hip
        return desired_hip

    def reset(self):
        # --- low‑level MuJoCo reset ---
        mujoco.mj_resetData(self.model, self.data)

        # --- domain randomisation ---------------------------------------
        # 1.  ramp tilt (rotate gravity about +y)
        tilt_deg = np.random.uniform(0.0, RAMP_DEG_MAX)
        tilt_rad = np.deg2rad(tilt_deg)
        self.model.opt.gravity[0] = 9.81 * np.sin(tilt_rad)
        self.model.opt.gravity[1] = 0.0
        self.model.opt.gravity[2] = -9.81 * np.cos(tilt_rad)

        # 2.  uniform geom friction
        friction_val = np.random.uniform(*FRICTION_MIN_MAX)
        # geom_friction is (ngeom, 3) → set the first coefficient
        self.model.geom_friction[:, 0] = friction_val

        # 3.  torso‑mass jitter ±10 %
        torso_body_id = mujoco.mj_name2id(self.model,
                                          mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.model.body_mass[torso_body_id] *= np.random.uniform(
            1.0 - MASS_JITTER, 1.0 + MASS_JITTER
        )

        # --- controller warm‑start & FSM reset --------------------------
        self.demo_hip = self.init_controller()
        mujoco.mj_forward(self.model, self.data)

        self.data.time  = 0.0
        self.fsm_hip    = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1  = self.FSM_KNEE1_STANCE
        self.fsm_knee2  = self.FSM_KNEE2_STANCE

        if self.randomize_legs:
            # Randomize the initial positions of the legs.
            self.data.qpos[self.hip_pos_actuator_id] = np.random.uniform(*self.hip_init_range)
            #self.data.qpos[self.left_knee_pos_actuator_id] = np.random.uniform(*self.knee_init_range)
            #self.data.qpos[self.right_knee_pos_actuator_id] = np.random.uniform(*self.knee_init_range)

            
        # --- bootstrap reward helpers -----------------------------------
        self.prev_x = float(self.data.qpos[0])         # forward progress ref
        # pitch (rad) absolute for upright bonus / fall check
        quat = self.data.xquat[torso_body_id]
        _, pitch, _ = quat2euler(quat)
        self.prev_pitch_abs = abs(pitch)

        # observation ready
        return self._get_obs()
    
    def _get_obs(self):
        torso_id = mujoco.mj_name2id(self.model,
                                     mujoco.mjtObj.mjOBJ_BODY, "torso")

        # base positions / velocities
        x      = float(self.data.qpos[0])
        z      = float(self.data.xpos[torso_id, 2])
        xdot   = float(self.data.qvel[0])
        zdot   = float(self.data.cvel[torso_id, 2])

        # torso pitch
        quat = self.data.xquat[torso_id]               # [w,x,y,z]
        _, pitch, _ = quat2euler(quat)

        # joint states
        hip_q   = float(self.data.qpos[self.hip_pos_actuator_id])
        lk_q    = float(self.data.qpos[self.left_knee_pos_actuator_id])
        rk_q    = float(self.data.qpos[self.right_knee_pos_actuator_id])
        hip_qd  = float(self.data.qvel[self.hip_pos_actuator_id])
        lk_qd   = float(self.data.qvel[self.left_knee_pos_actuator_id])
        rk_qd   = float(self.data.qvel[self.right_knee_pos_actuator_id])

        return np.array(
            [x, z, pitch,
             xdot, zdot,
             hip_q, lk_q, rk_q,
             hip_qd, lk_qd, rk_qd],
            dtype=np.float32
        )

    
    def controller_fsm_hip(self):
        """
        FSM logic for the hip joint only.
        Computes transitions based on the states of the legs and sets the hip actuator.
        """
        quat_left = self.data.xquat[self.left_leg_body_id, :]
        euler_left = quat2euler(quat_left)
        abs_left = -euler_left[1]
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        quat_right = self.data.xquat[self.right_leg_body_id, :]
        euler_right = quat2euler(quat_right)
        abs_right = -euler_right[1]
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        if self.fsm_hip == self.FSM_HIP_LEG2_SWING and pos_rightFoot[2] < 0.05 and abs_left < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG1_SWING
        elif self.fsm_hip == self.FSM_HIP_LEG1_SWING and pos_leftFoot[2] < 0.05 and abs_right < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG2_SWING

        if self.fsm_hip == self.FSM_HIP_LEG1_SWING:
            self.data.ctrl[self.hip_pos_actuator_id] = -0.5
        else:
            self.data.ctrl[self.hip_pos_actuator_id] = 0.5

    def controller_fsm_knees(self):
        """
        FSM logic for the knee joints only.
        """
        quat_leftLeg = self.data.xquat[self.left_leg_body_id, :]
        euler_leftLeg = quat2euler(quat_leftLeg)
        abs_leftLeg = -euler_leftLeg[1]
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        quat_rightLeg = self.data.xquat[self.right_leg_body_id, :]
        euler_rightLeg = quat2euler(quat_rightLeg)
        abs_rightLeg = -euler_rightLeg[1]
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        if self.fsm_knee1 == self.FSM_KNEE1_STANCE and pos_rightFoot[2] < 0.05 and abs_leftLeg < 0.0:
            self.fsm_knee1 = self.FSM_KNEE1_RETRACT
        elif self.fsm_knee1 == self.FSM_KNEE1_RETRACT and abs_leftLeg > 0.1:
            self.fsm_knee1 = self.FSM_KNEE1_STANCE

        if self.fsm_knee2 == self.FSM_KNEE2_STANCE and pos_leftFoot[2] < 0.05 and abs_rightLeg < 0.0:
            self.fsm_knee2 = self.FSM_KNEE2_RETRACT
        elif self.fsm_knee2 == self.FSM_KNEE2_RETRACT and abs_rightLeg > 0.1:
            self.fsm_knee2 = self.FSM_KNEE2_STANCE

        if self.fsm_knee1 == self.FSM_KNEE1_STANCE:
            self.data.ctrl[self.left_knee_pos_actuator_id] = 0.0
        else:
            self.data.ctrl[self.left_knee_pos_actuator_id] = -0.25

        if self.fsm_knee2 == self.FSM_KNEE2_STANCE:
            self.data.ctrl[self.right_knee_pos_actuator_id] = 0.0
        else:
            self.data.ctrl[self.right_knee_pos_actuator_id] = -0.25

    def step(self, external_action):
        """
        Advance the simulation by one time slice with shaped reward and
        early termination on fall.
        """
        sim_steps = int((1.0 / 60.0) / self.model.opt.timestep)

        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)

            # NN‑controlled knees
            if self.use_nn_for_knees:
                self.data.ctrl[self.left_knee_pos_actuator_id]  = external_action[1]
                self.data.ctrl[self.right_knee_pos_actuator_id] = external_action[2]
            else:
                self.controller_fsm_knees()

            # NN‑controlled hip
            if self.use_nn_for_hip:
                self.data.ctrl[self.hip_pos_actuator_id] = external_action[0]
            else:
                self.controller_fsm_hip()


        # -------- compute shaped reward ---------------------------------
        x_now = float(self.data.qpos[0])
        dx    = x_now - self.prev_x

        torso_id = mujoco.mj_name2id(self.model,
                                     mujoco.mjtObj.mjOBJ_BODY, "torso")
        quat = self.data.xquat[torso_id]
        _, pitch, _ = quat2euler(quat)
        pitch_abs = abs(pitch)

        upright_bonus = 0.5 if pitch_abs < UPRIGHT_PITCH_MAX else 0.0
        act_cost      = ACT_COST_COEFF * float(np.sum(np.abs(external_action)))

        # foot clearance
        foot_z = np.array([
            self.data.xpos[self.left_foot_body_id,  2],
            self.data.xpos[self.right_foot_body_id, 2]
        ])
        foot_clear = 0.2 if np.all(foot_z > FOOT_CLEAR_THRESH) else 0.0

        reward = dx + upright_bonus + foot_clear - act_cost

        # -------- fall detection & episode termination -------------------
        fallen = (self.data.xpos[torso_id, 2] < FALL_Z_MIN) or \
                 (pitch_abs > FALL_PITCH_MAX)
        if fallen:
            reward -= 5.0

        done = fallen or (self.data.time >= self.simend)

        # -------- update refs for next step ------------------------------
        self.prev_x         = x_now
        self.prev_pitch_abs = pitch_abs

        obs  = self._get_obs()
        info = {"time": self.data.time,
                "tilt_deg": np.rad2deg(pitch_abs)}

        return obs, reward, done, info

    def render(self, mode="human"):
        if not self.use_gui:
            return
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        self.cam.lookat[0] = self.data.qpos[0]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        if mode == "rgb_array":
            pass

    def close(self):
        if self.use_gui and self.window is not None:
            glfw.destroy_window(self.window)
            glfw.terminate()

if __name__ == "__main__":
    # Testing the updated environment.
    import os
    dirname = os.path.dirname(__file__)
    xml_path = os.path.join(dirname, "passiveWalker_model.xml")
    
    # Test demo mode (FSM for both hip and knees) with GUI.
    print("Testing FSM mode (demo) with GUI:")
    env_demo = PassiveWalkerEnv(xml_path, simend=10, use_nn_for_hip=False, use_nn_for_knees=False, use_gui=True, randomize_legs=True)
    obs = env_demo.reset()
    done = False
    total_reward = 0.0
    while not done and not glfw.window_should_close(env_demo.window):
        obs, reward, done, info = env_demo.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        total_reward += reward
        print(f"Time: {info['time']:.3f} | Reward: {reward:.3f} | Total: {total_reward:.3f}")
        env_demo.render(mode="human")
    env_demo.close()
    print("Demo mode with GUI finished.\n")
