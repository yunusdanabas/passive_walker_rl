"""
VLL Passive Walker Research Environment

Advanced environment with domain randomization, shaped rewards, and fall detection.
Supports both FSM and neural network control with comprehensive reward shaping.
"""

import os
import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw
import jax
import jax.numpy as jnp
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from constants import XML_PATH
from utils.control import (
    denormalize_action,
    compute_pd_control,
    get_joint_ranges,
    get_pd_gains,
    get_ctrl_limits,
)
from utils.jax_control import compute_pd_control_jax, compute_reward_jax, quat2euler_zyx_jit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class WalkerCfg:
    """Configuration parameters for the research environment."""
    ctrl_hz: int = 200  # Increased for stability
    upright_pitch: float = 0.20
    fall_z_min: float = 0.15
    fall_pitch_max: float = 1.0
    act_cost: float = 1e-3
    foot_clear: float = 0.03
    ramp_deg_max: float = 14.0
    ramp_deg_min: float = 10.0
    friction: tuple[float, float] = (0.8, 1.0)
    mass_jitter: float = 0.05

# FSM thresholds (hoisted to avoid repeated allocations)
CONTACT_HEIGHT = 0.05  # Contact detection height threshold
KNEE_RELEASE_THRESHOLD = 0.1  # Knee release threshold

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

# Target positions for PD control
HIP_SWING_POS = 0.5    # Hip forward swing (radians)
HIP_SWING_NEG = -0.5   # Hip backward swing (radians)
KNEE_STANCE = 0.0      # Knee stance phase (meters)
KNEE_RETRACT = -0.25   # Knee retraction phase (meters)


# Reward coefficients
REWARD_COEFFS = {
    'forward_progress': 1.0,
    'upright_bonus': 0.5,
    'foot_clear': 0.2,
    'action_cost': 0.0015,
}

# ============================================================================
# Reward Configuration for RL Training
# ============================================================================

@dataclass
class RewCfg:
    """Reward configuration for RL training environment."""
    # Core terms
    c_fp: float = 1.0                    # Forward progress coefficient
    c_up: float = 0.5                    # Upright bonus coefficient
    upright_pitch_max: float = 0.25      # Max pitch for upright bonus (rad)
    c_ac: float = 3e-4                   # Action cost coefficient
    
    # Optional terms
    c_vt: float = 0.25                   # Velocity tracking coefficient
    vx_star: float = 0.8                 # Target velocity (m/s)
    sigma_v: float = 0.25                # Velocity tracking width
    
    c_sym: float = 0.05                  # Symmetry coefficient
    sigma_sym: float = 0.4               # Symmetry width
    
    c_fc: float = 0.05                   # Foot clearance coefficient
    foot_clear_target: float = 0.03      # Target foot clearance (m)
    
    # Terminal conditions
    pen_fall: float = 5.0                # Fall penalty
    fall_pitch_max: float = 1.0          # Max pitch before fall (rad)
    fall_z_min: float = 0.15             # Min torso height before fall (m)
    
    # Reward clipping
    clip_low: float = -5.0               # Lower reward bound
    clip_high: float = 5.0               # Upper reward bound

# ============================================================================
# Utility Functions
# ============================================================================

def _is_fallen(xpos, pitch, cfg, torso_id):
    """
    Check if walker has fallen based on height and pitch criteria.
    
    Args:
        xpos: Position array from MuJoCo
        pitch: Pitch angle from pitch joint (radians)
        cfg: WalkerCfg instance with fall thresholds
        torso_id: Torso body ID
        
    Returns:
        bool: True if walker has fallen
    """
    torso_z = xpos[torso_id, 2]
    return (torso_z < cfg.fall_z_min) or (abs(pitch) > cfg.fall_pitch_max)


def quat2euler(q):
    """
    Convert MuJoCo quaternion [w,x,y,z] to roll-pitch-yaw (radians) using MuJoCo's optimized functions.
    
    Args:
        q: Quaternion array [w, x, y, z]
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Convert to MuJoCo format (column vector)
    quat_mj = q.reshape(4, 1)
    
    # Convert quaternion to rotation matrix using MuJoCo's optimized function
    mat = np.zeros((9, 1))
    mujoco.mju_quat2Mat(mat, quat_mj)
    
    # Reshape to 3x3 matrix
    R = mat.reshape(3, 3)
    
    # Extract Euler angles using ZYX convention (yaw-pitch-roll)
    # This matches the aerospace convention used in robotics
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])   # x-axis rotation
        pitch = np.arctan2(-R[2,0], sy)     # y-axis rotation  
        yaw = np.arctan2(R[1,0], R[0,0])    # z-axis rotation
    else:
        roll = np.arctan2(-R[1,2], R[1,1])  # x-axis rotation
        pitch = np.arctan2(-R[2,0], sy)     # y-axis rotation
        yaw = 0                             # z-axis rotation

    return roll, pitch, yaw



def _randomise_physics(self, rng):
    """Apply domain randomization to physics parameters using NumPy (faster than JAX)."""
    # Save defaults
    default_gravity = tuple(self.model.opt.gravity)
    default_friction = float(self.model.geom_friction[0, 0])
    torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    default_mass = float(self.model.body_mass[torso_body_id])

    # Randomize ramp tilt using NumPy (faster than JAX for single values)
    tilt_deg = np.random.uniform(self.cfg.ramp_deg_min, self.cfg.ramp_deg_max)
    tilt_rad = np.deg2rad(tilt_deg)
    new_gravity = (
        9.81 * np.sin(tilt_rad),
        0.0,
        -9.81 * np.cos(tilt_rad)
    )
    self.model.opt.gravity[0] = new_gravity[0]
    self.model.opt.gravity[1] = new_gravity[1]
    self.model.opt.gravity[2] = new_gravity[2]

    # Randomize friction using NumPy
    friction = np.random.uniform(self.cfg.friction[0], self.cfg.friction[1])
    self.model.geom_friction[:, 0] = friction

    # Randomize torso mass using NumPy
    mass_jit = np.random.uniform(1-self.cfg.mass_jitter, 1+self.cfg.mass_jitter)
    new_mass = default_mass * mass_jit
    self.model.body_mass[torso_body_id] = new_mass

    return rng


class PassiveWalkerEnv(gym.Env):
    """
    VLL Passive Walker Research Environment
    
    Features: Domain randomization, shaped rewards, fall detection
    Action API: 3D [hip, left_knee, right_knee] in [-1, 1] mapping to physical ranges via denorm
    Observation: 11D [x, z, pitch, ẋ, ż, hip_q, lk_q, rk_q, hip_q̇, lk_q̇, rk_q̇]
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
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        
        # Define observation space to match _get_obs() output
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        # Define action space for the 3 actuators (hip and two knees)
        # Actions are normalized [-1, 1] and denormalized to physical ranges via denormalize_action()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Pre-allocate observation buffer
        self.obs_buf = np.empty(11, dtype=np.float32)
        
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

        # Retrieve IDs.
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.hip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hip")
        self.left_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_leg")
        self.right_leg_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_leg")
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        self.hip_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_act")
        self.left_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_knee_act")
        self.right_knee_pos_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_knee_act")
        
        # Get joint IDs for knees
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
        
        # Pre-allocate observation buffer for zero-allocation observations
        self.obs_buf = np.empty(11, dtype=np.float32)
        
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



    def reset(self):
        # --- low‑level MuJoCo reset ---
        mujoco.mj_resetData(self.model, self.data)

        # --- domain randomisation ---------------------------------------
        if self.randomize_physics:
            self.rng, sub = jax.random.split(self.rng)
            _randomise_physics(self, sub)

        # --- Reset FSM states first -------------------------------------
        self.fsm_hip    = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1  = self.FSM_KNEE1_STANCE
        self.fsm_knee2  = self.FSM_KNEE2_STANCE

        # --- Set initial joint states using joint addresses -------------
        self.data.qpos[self.hip_qpos_addr] = HIP_SWING_NEG
        self.data.qpos[self.left_knee_qpos_addr] = 0.0
        self.data.qpos[self.right_knee_qpos_addr] = 0.0

        # --- Zero all control signals (torques/forces) -----------------
        self.data.ctrl[self.hip_pos_actuator_id] = 0.0
        self.data.ctrl[self.left_knee_pos_actuator_id] = 0.0
        self.data.ctrl[self.right_knee_pos_actuator_id] = 0.0

        # --- Propagate qpos→xpos, xquat, etc ---------------------------
        mujoco.mj_forward(self.model, self.data)

        # --- Reset timer ------------------------------------------------
        self.data.time = 0.0

        # --- bootstrap reward helpers -----------------------------------
        self.prev_x = float(self.data.qpos[self.slide_x_dof])  # forward progress ref
        # pitch (rad) absolute for upright bonus / fall check (use pitch joint directly)
        pitch = float(self.data.qpos[self.pitch_qpos_addr])
        self.prev_pitch_abs = abs(pitch)
        
        # Initialize applied controls for reward calculation
        self.last_applied_controls = np.zeros(3, dtype=np.float32)

        # observation ready
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
        ob = self.obs_buf
        
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
        # Get current joint states
        q_current = np.array([
            self.data.qpos[self.hip_qpos_addr],
            self.data.qpos[self.left_knee_qpos_addr], 
            self.data.qpos[self.right_knee_qpos_addr]
        ])
        qd_current = np.array([
            self.data.qvel[self.hip_qvel_addr],
            self.data.qvel[self.left_knee_qvel_addr],
            self.data.qvel[self.right_knee_qvel_addr]
        ])
        
        # Determine desired positions based on control mode
        q_desired = np.zeros(3)
        
        # Hip control
        if self.use_nn_for_hip:
            # Denormalize action to physical range
            joint_ranges = get_joint_ranges()
            q_desired[0] = denormalize_action([external_action[0]], [joint_ranges[0]])[0]
        else:
            # Use FSM logic to determine desired position (without setting control)
            self._update_fsm_hip_state()
            if self.fsm_hip == self.FSM_HIP_LEG1_SWING:
                q_desired[0] = HIP_SWING_NEG  # FSM sets -0.5
            else:
                q_desired[0] = HIP_SWING_POS   # FSM sets 0.5
        
        # Knee control
        if self.use_nn_for_knees:
            # Denormalize actions to physical ranges
            joint_ranges = get_joint_ranges()
            q_desired[1] = denormalize_action([external_action[1]], [joint_ranges[1]])[0]
            q_desired[2] = denormalize_action([external_action[2]], [joint_ranges[2]])[0]
        else:
            # Use FSM logic to determine desired positions (without setting control)
            self._update_fsm_knee_states()
            q_desired[1] = KNEE_STANCE if self.fsm_knee1 == self.FSM_KNEE1_STANCE else KNEE_RETRACT
            q_desired[2] = KNEE_STANCE if self.fsm_knee2 == self.FSM_KNEE2_STANCE else KNEE_RETRACT
        
        # Compute PD control using NumPy (faster for small vectors)
        controls = pd_step_np(q_current, qd_current, q_desired, self._kp, self._kv, self._umin, self._umax)
        
        # Apply controls to actuators
        self.data.ctrl[self.hip_pos_actuator_id] = controls[0]
        self.data.ctrl[self.left_knee_pos_actuator_id] = controls[1]
        self.data.ctrl[self.right_knee_pos_actuator_id] = controls[2]
        
        # Store applied controls for reward calculation
        self.last_applied_controls = controls.copy()

    def step(self, external_action):
        """
        Advance the simulation by one time slice with shaped reward and
        early termination on fall.
        """
        sim_steps = max(1, int((1.0 / self.ctrl_hz) / self.model.opt.timestep))
        
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
        self.last_applied_controls[:] = u

        # Calculate reward components (using prebound variables)
        current_x = qpos[self.slide_x_dof]
        dx = current_x - self.prev_x  # Forward progress
        
        # Get additional reward components
        pitch = float(qpos[self.pitch_qpos_addr])
        pitch_abs = abs(pitch)
        ctrl_abs_sum = np.sum(np.abs(u))  # Use current controls
        
        # Get torso position for fall detection
        torso_z = data.xpos[self.torso_body_id, 2]
        
        # Get foot positions for clearance bonus
        left_foot_z = data.xpos[self.left_foot_body_id, 2]
        right_foot_z = data.xpos[self.right_foot_body_id, 2]
        
        # Compute smooth, RL-friendly rewards
        reward_cfg = RewCfg()  # Use default configuration
        
        # 1. Forward progress (dense)
        forward_reward = reward_cfg.c_fp * dx
        
        # 2. Smooth upright bonus (no hard thresholds)
        upright_ratio = pitch_abs / reward_cfg.upright_pitch_max
        upright_bonus = reward_cfg.c_up * max(0.0, 1.0 - upright_ratio**2)
        
        # 3. Action cost (L1 norm)
        action_cost = reward_cfg.c_ac * ctrl_abs_sum
        
        # 4. Optional: Velocity tracking (encourage target speed)
        vx = qvel[self.slide_x_dof]
        vel_diff = (vx - reward_cfg.vx_star) / reward_cfg.sigma_v
        vel_tracking = reward_cfg.c_vt * np.exp(-vel_diff**2 / 2.0)
        
        # 5. Optional: Symmetry (penalize left-right bias)
        lk_q = qpos[self.left_knee_qpos_addr]
        rk_q = qpos[self.right_knee_qpos_addr]
        knee_diff = (lk_q - rk_q) / reward_cfg.sigma_sym
        symmetry = reward_cfg.c_sym * np.exp(-knee_diff**2 / 2.0)
        
        # 6. Optional: Soft foot clearance (smooth instead of hard threshold)
        def soft_hinge(x):
            return np.log1p(np.exp(x))  # smooth approximation of max(0, x)
        
        left_clear = soft_hinge(left_foot_z - reward_cfg.foot_clear_target)
        right_clear = soft_hinge(right_foot_z - reward_cfg.foot_clear_target)
        foot_clearance = reward_cfg.c_fc * 0.5 * (left_clear + right_clear)
        
        # 7. Fall detection and penalty
        fell = (pitch_abs > reward_cfg.fall_pitch_max) or (torso_z < reward_cfg.fall_z_min)
        fall_penalty = reward_cfg.pen_fall if fell else 0.0
        
        # 8. Total reward (smooth, dense, clipped)
        raw_reward = (forward_reward + upright_bonus + vel_tracking + 
                     symmetry + foot_clearance - action_cost - fall_penalty)
        reward = np.clip(raw_reward, reward_cfg.clip_low, reward_cfg.clip_high)

        # Episode termination
        done = fell or (data.time >= self.simend)

        # Update reference for next step
        self.prev_x = current_x
        self.prev_pitch_abs = pitch_abs

        obs = self._get_obs()
        info = {
            "time": data.time,
            "tilt_deg": np.rad2deg(pitch_abs)
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
    import time
    
    # Configuration for printing
    PRINT_ENABLED = True  # Set to False to disable all printing
    PRINT_INTERVAL = 2.0  # Print every 2 seconds (less frequent)
    
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
    if PRINT_ENABLED:
        print("Testing FSM mode (demo) with GUI:")
    
    env_demo = PassiveWalkerEnv(
        xml_path, 
        simend=30, 
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
    
    # Throttle printing (configurable interval)
    t0 = time.time()
    last_print = t0
    n = 0
    
    while not done and (not env_demo.window or not glfw.window_should_close(env_demo.window)):
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
