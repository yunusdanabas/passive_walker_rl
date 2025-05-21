# passive_walker/envs/mujoco_env.py
# Passive Walker Environment using MuJoCo and Gym

import os
import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw
import jax, jax.numpy as jnp
import logging
from dataclasses import dataclass

from passive_walker.constants import XML_PATH

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class WalkerCfg:
    ctrl_hz: int = 200
    upright_pitch: float = 0.20
    fall_z_min: float = 0.15
    fall_pitch_max: float = 1.0
    act_cost: float = 1e-3
    foot_clear: float = 0.03
    ramp_deg_max: float = 14.0
    ramp_deg_min: float = 10.0
    friction: tuple[float, float] = (0.8, 1.0)
    mass_jitter: float = 0.05

# ---------- Constants --------------------------------------------------
# Reward coefficients
REWARD_COEFFS = {
    'forward_progress': 1.0,      # Reasonable weight for forward movement
    'upright_bonus': 0.5,          # Bonus for staying upright
    'foot_clear': 0.2,            # Bonus for foot clearance
    'action_cost': 0.1            # Small penalty for actions
}

# ---------- Helper:  fallen? ----------------------------------------------
def _is_fallen(xpos, rot, cfg, torso_id):
    """
    Returns True if torso is too low or pitch exceeds limits.
    Uses quat2euler to obtain pitch.
    """
    _, pitch, _ = quat2euler(rot)        # rot = [w,x,y,z]
    torso_z     = xpos[torso_id, 2]      # use actual torso_id
    return (torso_z < cfg.fall_z_min) or (abs(pitch) > cfg.fall_pitch_max)


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

# ---------- Helper:  shaped reward ----------------------------------------
def _compute_reward(xpos, rot, act, metrics, cfg):
    """
    Compute the reward using JAX operations.
    Args:
        xpos: Position array from MuJoCo
        rot: Rotation (quaternion) array from MuJoCo
        act: Action vector
        metrics: Dictionary containing previous state metrics
        cfg: WalkerCfg instance for reward parameters
    Returns:
        Total reward as a JAX array
    """
    x_now = xpos[0, 0]
    # Negate dx because negative x is forward in MuJoCo
    dx = -(x_now - metrics['prev_x'])            # forward progress

    # upright bonus
    upright_bonus = jnp.where(metrics['pitch_abs'] < cfg.upright_pitch, 
                             REWARD_COEFFS['upright_bonus'], 0.0)

    # action penalty (act is 3‑D scaled action ∈[‑1,1])
    act_cost = REWARD_COEFFS['action_cost'] * jnp.sum(jnp.abs(act))

    # foot‑drag bonus
    foot_z = xpos[1:, 2]                      # both feet
    foot_clear = jnp.where(jnp.all(foot_z > cfg.foot_clear), 
                          REWARD_COEFFS['foot_clear'], 0.0)

    # Combine rewards with coefficients
    forward_reward = REWARD_COEFFS['forward_progress'] * dx
    
    total_reward = (forward_reward + 
                   upright_bonus + 
                   foot_clear - 
                   act_cost)
    
    # Debug information
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Reward components:")
        logger.debug(f"  Forward progress (dx={dx:.3f}): {forward_reward:.3f}")
        logger.debug(f"  Upright bonus: {upright_bonus:.3f}")
        logger.debug(f"  Foot clearance: {foot_clear:.3f}")
        logger.debug(f"  Action cost: {act_cost:.3f}")
        logger.debug(f"  Total reward: {total_reward:.3f}")
    
    return total_reward


# ---------- Helper:  apply domain randomisation in reset -------------------
def _randomise_physics(self, rng):
    """
    Apply domain randomization to physics parameters using JAX.
    Args:
        rng: JAX random key
    Returns:
        Updated random key
    """
    # 1. Randomize ramp tilt (rotate gravity about +y)
    rng, sub = jax.random.split(rng)
    tilt_deg = jax.random.uniform(sub, (), minval=self.cfg.ramp_deg_min, maxval=self.cfg.ramp_deg_max)
    tilt_rad = jnp.deg2rad(tilt_deg)
    # Only modify gravity components, preserving y-component
    self.model.opt.gravity[0] = 9.81 * np.sin(tilt_rad)
    self.model.opt.gravity[1] = 0.0
    self.model.opt.gravity[2] = -9.81 * np.cos(tilt_rad)

    # 2. Randomize friction
    rng, sub = jax.random.split(rng)
    friction = float(jax.random.uniform(sub, (), 
                                minval=self.cfg.friction[0],
                                maxval=self.cfg.friction[1]))
    self.model.geom_friction[:, 0] = friction

    # 3. Randomize torso mass
    rng, sub = jax.random.split(rng)
    mass_jit = float(jax.random.uniform(sub, (), 
                                minval=1-self.cfg.mass_jitter, 
                                maxval=1+self.cfg.mass_jitter))
    torso_body_id = mujoco.mj_name2id(self.model,
                                     mujoco.mjtObj.mjOBJ_BODY, "torso")
    original_mass = self.model.body_mass[torso_body_id]
    self.model.body_mass[torso_body_id] *= mass_jit

    # Debug printing
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("\nPhysics Randomization:")
        logger.debug(f"  Ramp tilt: {float(tilt_deg):.1f}°")
        logger.debug(f"  Gravity vector: [{self.model.opt.gravity[0]:.2f}, {self.model.opt.gravity[1]:.2f}, {self.model.opt.gravity[2]:.2f}]")
        logger.debug(f"  Friction coefficient: {friction:.2f}")
        logger.debug(f"  Torso mass: {original_mass:.2f} → {self.model.body_mass[torso_body_id]:.2f} (×{mass_jit:.2f})")
        logger.debug("")

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
    
    def __init__(self, xml_path=XML_PATH, 
                 simend=30.0, 
                 use_nn_for_hip=False, 
                 use_nn_for_knees=False,
                 use_gui=True, 
                 cfg: WalkerCfg = WalkerCfg(),
                 rng_seed: int = None,
                 randomize_physics: bool = False):
        
        super().__init__()
        self.simend = simend
        self.use_nn_for_hip = use_nn_for_hip
        self.use_nn_for_knees = use_nn_for_knees
        self.use_gui = use_gui
        self.cfg = cfg
        self.ctrl_hz = cfg.ctrl_hz
        self.randomize_physics = randomize_physics
        # Use time-based seed if none provided
        if rng_seed is None:
            rng_seed = int(np.random.randint(2**31))
        self.rng = jax.random.PRNGKey(rng_seed)

        # Initialize window as None, will be created when needed
        self.window = None
        self.cam = None
        self.opt = None
        self.scene = None
        self.context = None

        # Load the MuJoCo model and simulation data.
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define observation space to match _get_obs() output
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        # Define action space for the 3 actuators (hip and two knees)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Pre-allocate observation buffer
        self.obs_buf = np.empty(11, dtype=np.float32)

        # Retrieve IDs.
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        self.left_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_leg")
        self.right_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

        self.hip_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act")
        self.left_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.right_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        
        # Get joint IDs for knees
        self.left_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.right_knee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        
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
        
    def _ensure_window(self):
        """Create the GLFW window and MuJoCo visualization objects when needed."""
        if self.window is None:
            if not glfw.init():
                raise Exception("GLFW initialization failed")
            self.window = glfw.create_window(1200, 900, "Passive Walker Viewer", None, None)
            if not self.window:
                glfw.terminate()
                raise Exception("GLFW window creation failed")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            
            # Initialize visualization objects
            self.cam = mujoco.MjvCamera()
            self.cam.distance = 8.0
            self.opt = mujoco.MjvOption()
            self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
            self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

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
        if self.randomize_physics:
            self.rng, sub = jax.random.split(self.rng)
            _randomise_physics(self, sub)

        # --- controller warm‑start & FSM reset --------------------------
        self.demo_hip = self.init_controller()
        mujoco.mj_forward(self.model, self.data)

        self.data.time  = 0.0
        self.fsm_hip    = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1  = self.FSM_KNEE1_STANCE
        self.fsm_knee2  = self.FSM_KNEE2_STANCE

        # --- bootstrap reward helpers -----------------------------------
        self.prev_x = float(self.data.qpos[0])         # forward progress ref
        # pitch (rad) absolute for upright bonus / fall check
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        quat = self.data.xquat[torso_id]
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

        # joint states using joint IDs
        hip_q   = float(self.data.qpos[self.hip_id])
        lk_q    = float(self.data.qpos[self.left_knee_id])
        rk_q    = float(self.data.qpos[self.right_knee_id])
        hip_qd  = float(self.data.qvel[self.hip_id])
        lk_qd   = float(self.data.qvel[self.left_knee_id])
        rk_qd   = float(self.data.qvel[self.right_knee_id])

        # Fill observation buffer
        ob = self.obs_buf
        ob[:] = [x, z, pitch,
                 xdot, zdot,
                 hip_q, lk_q, rk_q,
                 hip_qd, lk_qd, rk_qd]
        return ob

    
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
        sim_steps = int((1.0 / self.ctrl_hz) / self.model.opt.timestep)
        # print(f"sim_steps: {sim_steps}")
        # print(f"self.ctrl_hz: {self.ctrl_hz}")
        # print(f"self.model.opt.timestep: {self.model.opt.timestep}")

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

        # Convert MuJoCo data to JAX arrays for reward computation
        xpos = jnp.asarray(self.data.xpos)
        rot = jnp.asarray(self.data.xquat)
        
        metrics = {
            'prev_x': self.prev_x,
            'pitch_abs': self.prev_pitch_abs
        }
        
        # Compute reward using JAX
        reward = float(_compute_reward(xpos, rot, jnp.asarray(external_action), metrics, self.cfg))

        # -------- fall detection & episode termination -------------------
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        fallen   = _is_fallen(self.data.xpos,
                            self.data.xquat[torso_id],
                            self.cfg,
                            torso_id)

        if fallen:
            reward -= 5.0

        done = fallen or (self.data.time >= self.simend)

        # -------- update refs for next step ------------------------------
        self.prev_x = float(self.data.qpos[0])
        _, pitch, _ = quat2euler(self.data.xquat[torso_id])
        self.prev_pitch_abs = abs(pitch)

        obs = self._get_obs()
        info = {
            "time": self.data.time,
            "tilt_deg": np.rad2deg(abs(pitch))
        }

        return obs, reward, done, info

    def render(self, mode="human"):
        if not self.use_gui:
            return
        
        self._ensure_window()
            
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
    # Path to your MuJoCo XML
    xml_path = str(XML_PATH)
    
    # Create config with default parameters
    cfg = WalkerCfg(
        ctrl_hz=60,
        ramp_deg_max=11.5,
        friction=(0.8, 1.0),
        mass_jitter=0.05,
    )
    
    # Test demo mode (FSM for both hip and knees) with GUI.
    print("Testing FSM mode (demo) with GUI:")
    env_demo = PassiveWalkerEnv(
        xml_path, 
        simend=15, 
        use_nn_for_hip=False, 
        use_nn_for_knees=False, 
        use_gui=True, 
        cfg=cfg,
        rng_seed=0,
        randomize_physics=True
    )
    
    # Enable debug logging if desired
    # logger.setLevel(logging.DEBUG)
    
    obs = env_demo.reset()
    done = False
    total_reward = 0.0
    while not done and (not env_demo.window or not glfw.window_should_close(env_demo.window)):
        obs, reward, done, info = env_demo.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        total_reward += reward
        print(f"Time: {info['time']:.3f} | Reward: {reward:.3f} | Total: {total_reward:.3f}")
        env_demo.render(mode="human")
    env_demo.close()
    print("Demo mode with GUI finished.\n")
