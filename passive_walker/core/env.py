"""
Passive Walker Environment

Bipedal walking environment with FSM and neural network control modes.
Supports both headless and GUI operation with physics simulation.
"""
from __future__ import annotations
from typing import Optional

# =====================
# Configuration
# =====================
# Simulation parameters
CTRL_HZ = 100.0          # Controller frequency (Hz)
SIM_SECONDS = 25.0      # Default episode length (s)
XML_PATH = "passive_walker/assets/passiveWalker_model.xml"

# Physics parameters
RAMP_DEG = 10.0         # Incline angle (degrees, positive = downhill)
FRICTION = 0.9          # Contact friction coefficient
RANDOMIZE_PHYSICS = False  # Enable domain randomization
MASS_JITTER = 0.05      # ±5% torso mass variation when randomization enabled
RAMP_JITTER = 0.0       # Ramp angle variation (degrees, ±range when randomization enabled)
FRICTION_MIN = 0.6      # Minimum friction coefficient when randomization enabled
FRICTION_MAX = 1.0      # Maximum friction coefficient when randomization enabled

# Rendering parameters
DEFAULT_GUI = True      # Enable GUI when run as module
CAM_DISTANCE = 8.0      # Camera distance from walker

# Observation and action dimensions
_OBS_DIM = 17  # [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇, lc, rc, lf, rf, lcd, rcd]
# Original 11D: [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
# Contact 6D: [left_contact, right_contact, left_force, right_force, left_contact_duration, right_contact_duration]
_ACT_DIM = 3   # [hip, left_knee, right_knee] - knee sliders (m)


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco.glfw import glfw

from .controller import PDController, FSMStateMachine
from .reward import get_reward_fn
from .randomization import DomainRandomizer, RandomizationConfig, get_randomization_config


class PassiveWalkerEnv(gym.Env):
    """
    Passive Walker Environment with FSM and Neural Network control.
    
    Modes:
        - 'fsm': Uses built-in FSM for stable walking (actions ignored)
        - 'research': Uses neural network actions for all joints
        - 'hybrid_hip': Uses neural network for hip, FSM for knees
        - 'hybrid_knees': Uses FSM for hip, neural network for knees
    
    Args:
        mode: Control mode (default: "fsm")
        use_gui: Enable GUI rendering (default: False)
        use_jax_pd: Use JAX backend for PD control (default: False, uses NumPy)
        ramp_deg: Ramp angle in degrees (default: 10.0)
        friction: Contact friction coefficient (default: 0.9)
        randomize_physics: Enable domain randomization (default: False)
        ramp_jitter: Ramp angle variation in degrees when randomizing (default: 0.0)
        friction_min: Minimum friction when randomizing (default: 0.6)
        friction_max: Maximum friction when randomizing (default: 1.0)
        ctrl_hz: Control frequency in Hz (default: 100.0). Higher frequencies (e.g., 200Hz)
                 may improve stability but increase computational cost.
        randomization_profile: Use predefined advanced randomization profile 
                               ("none", "basic", "moderate", "aggressive", "temporal").
                               If None, uses basic randomization with above parameters.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, mode: str = "fsm", use_gui: bool = False, use_jax_pd: bool = False,
                 ramp_deg: float = None, friction: float = None, randomize_physics: bool = None,
                 ramp_jitter: float = None, friction_min: float = None, friction_max: float = None,
                 ctrl_hz: float = None, randomization_profile: str = None):
        super().__init__()
        assert mode in ("fsm", "research", "hybrid_hip", "hybrid_knees")
        self.mode = mode
        self.use_gui = use_gui
        self.ctrl_hz = float(ctrl_hz if ctrl_hz is not None else CTRL_HZ)
        self.simend = float(SIM_SECONDS)
        
        # Physics parameters (use provided values or defaults)
        self._base_ramp_deg = float(ramp_deg if ramp_deg is not None else RAMP_DEG)
        self._base_friction = float(friction if friction is not None else FRICTION)
        self.ramp_deg = self._base_ramp_deg
        self.friction = self._base_friction
        self.randomize_physics = bool(randomize_physics if randomize_physics is not None else RANDOMIZE_PHYSICS)
        self.ramp_jitter = float(ramp_jitter if ramp_jitter is not None else RAMP_JITTER)
        self.friction_min = float(friction_min if friction_min is not None else FRICTION_MIN)
        self.friction_max = float(friction_max if friction_max is not None else FRICTION_MAX)
        
        # Advanced randomization
        self.randomization_profile = randomization_profile
        self.domain_randomizer = None  # Initialized after model loading

        self.window = None
        self._np_rng = None  # Random number generator

        # Initialize MuJoCo model and body/joint IDs
        self._init_model_and_ids(XML_PATH)

        # Define observation and action spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACT_DIM,), dtype=np.float32)

        # Initialize controllers
        # PD Controller: NumPy (fast) vs JAX (vectorized) backend selection
        # NumPy is default for single-environment performance
        self.pd = PDController(use_jax=use_jax_pd)
        self.fsm = FSMStateMachine()
        
        # Bind FSM indices for performance (after model is loaded)
        self.fsm.bind_indices(
            qpos_hip=self.qpos_hip, qpos_lk=self.qpos_lk, qpos_rk=self.qpos_rk,
            qvel_hip=self.qvel_hip, qvel_lk=self.qvel_lk, qvel_rk=self.qvel_rk,
            b_lfoot=self.b_lfoot, b_rfoot=self.b_rfoot,
            b_lleg=self.b_lleg, b_rleg=self.b_rleg
        )

        # Initialize reward function
        self.reward_fn = get_reward_fn(self.mode)

        # Pre-allocate arrays for performance
        self._obs = np.empty(_OBS_DIM, dtype=np.float32)
        self._q = np.empty(3, dtype=np.float32)
        self._qd = np.empty(3, dtype=np.float32)
        self._qdes = np.zeros(3, dtype=np.float32)
        self._u = np.empty(3, dtype=np.float32)
        
        # Contact information tracking
        self._left_contact_duration = 0.0
        self._right_contact_duration = 0.0
        self._prev_left_contact = False
        self._prev_right_contact = False
        self._contact_threshold = 0.1  # Force threshold for contact detection

        self.prev_x = 0.0
        self._t_next = 0.0  # Target time for next control step
        self._seed_used = None  # Track seed for dataset provenance
        self._timing_debug = False  # Enable timing debug logging

        # GUI window will be created on first render() call (like legacy)

    def _init_model_and_ids(self, xml_path: str) -> None:
        """Load MuJoCo model and cache body/joint/actuator IDs."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Helper functions for ID lookup
        def jid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        def bid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        def aid(name): return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Joint IDs and DOF addresses
        self.j_slide_x = jid("slide_x")
        self.dof_x = self.model.jnt_dofadr[self.j_slide_x]
        self.qpos_x = self.model.jnt_qposadr[self.j_slide_x]
        self.j_slide_z = jid("slide_z")
        self.dof_z = self.model.jnt_dofadr[self.j_slide_z]
        self.qpos_z = self.model.jnt_qposadr[self.j_slide_z]
        self.j_pitch = jid("pitch")
        self.qpos_pitch = self.model.jnt_qposadr[self.j_pitch]
        self.qvel_pitch = self.model.jnt_dofadr[self.j_pitch]
        self.j_hip = jid("hip")
        self.qpos_hip = self.model.jnt_qposadr[self.j_hip]
        self.qvel_hip = self.model.jnt_dofadr[self.j_hip]
        self.j_lk = jid("left_knee")
        self.qpos_lk = self.model.jnt_qposadr[self.j_lk]
        self.qvel_lk = self.model.jnt_dofadr[self.j_lk]
        self.j_rk = jid("right_knee")
        self.qpos_rk = self.model.jnt_qposadr[self.j_rk]
        self.qvel_rk = self.model.jnt_dofadr[self.j_rk]

        # Body IDs
        self.b_torso = bid("torso")
        self.b_lfoot = bid("left_foot")
        self.b_rfoot = bid("right_foot")
        self.b_lleg = bid("left_leg")     # For FSM leg angle detection
        self.b_rleg = bid("right_leg")    # For FSM leg angle detection

        # Actuator IDs
        self.a_hip = aid("hip_act")
        self.a_lk = aid("left_knee_act")
        self.a_rk = aid("right_knee_act")

        # Apply base physics (gravity, friction)
        self._apply_base_physics()
        
        # Cache original torso mass for domain randomization
        self._torso_mass0 = float(self.model.body_mass[self.b_torso])
        
        # Initialize domain randomizer if profile specified
        if self.randomization_profile is not None:
            # Will be fully initialized in reset() when RNG is available
            self._randomization_config = get_randomization_config(self.randomization_profile)
        else:
            self._randomization_config = None

    def _apply_base_physics(self):
        """Set up base physics parameters (gravity, friction)."""
        tilt = np.deg2rad(self.ramp_deg)
        self.model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]
        self.model.geom_friction[:, 0] = float(self.friction)

    def _randomize_physics(self):
        """Apply domain randomization to physics parameters."""
        assert self._np_rng is not None
        
        # Use advanced randomization if profile is set
        if self._randomization_config is not None:
            if self.domain_randomizer is None:
                self.domain_randomizer = DomainRandomizer(
                    self._randomization_config, self.model, self._np_rng
                )
            self.ramp_deg, self.friction = self.domain_randomizer.randomize_all(
                self._base_ramp_deg, self._base_friction
            )
        else:
            # Basic randomization
            if self.ramp_jitter > 0:
                self.ramp_deg = self._np_rng.uniform(
                    self._base_ramp_deg - self.ramp_jitter,
                    self._base_ramp_deg + self.ramp_jitter
                )
            if self.friction_min < self.friction_max:
                self.friction = self._np_rng.uniform(self.friction_min, self.friction_max)
            
            # Randomize torso mass
            scale = self._np_rng.uniform(1.0 - MASS_JITTER, 1.0 + MASS_JITTER)
            self.model.body_mass[self.b_torso] = self._torso_mass0 * scale
        
        # Apply physics with randomized parameters
        self._apply_base_physics()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None or self._np_rng is None:
            self._np_rng = np.random.RandomState(None if seed is None else int(seed))
            self._seed_used = seed

        mujoco.mj_resetData(self.model, self.data)
        self.fsm.reset()

        # Set initial pose (hip at -0.5 to match legacy FSM)
        self.data.qpos[self.qpos_hip] = -0.5
        self.data.qpos[self.qpos_lk] = 0.0
        self.data.qpos[self.qpos_rk] = 0.0
        self.data.ctrl[self.a_hip] = self.data.ctrl[self.a_lk] = self.data.ctrl[self.a_rk] = 0.0

        # Apply physics (base or randomized)
        if self.randomize_physics:
            self._randomize_physics()
        else:
            self._apply_base_physics()

        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self.prev_x = float(self.data.qpos[self.qpos_x])
        self._t_next = 1.0 / self.ctrl_hz  # First control step target
        
        # Initialize control change tracking for smooth motion penalty
        self._prev_u = np.zeros(3, dtype=np.float32)
        
        # Reset contact tracking
        self._left_contact_duration = 0.0
        self._right_contact_duration = 0.0
        self._prev_left_contact = False
        self._prev_right_contact = False

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation vector."""
        qpos, qvel = self.data.qpos, self.data.qvel
        ob = self._obs
        
        # Original 11D state
        ob[0] = qpos[self.qpos_x]     # x position
        ob[1] = qpos[self.qpos_z]     # z position
        ob[2] = qpos[self.qpos_pitch] # pitch angle
        ob[3] = qvel[self.dof_x]      # x velocity
        ob[4] = qvel[self.dof_z]      # z velocity
        ob[5] = qpos[self.qpos_hip]   # hip angle
        ob[6] = qpos[self.qpos_lk]    # left knee slider (m)
        ob[7] = qpos[self.qpos_rk]    # right knee slider (m)
        ob[8] = qvel[self.qvel_hip]   # hip angular velocity
        ob[9] = qvel[self.qvel_lk]    # left knee slider velocity (m/s)
        ob[10] = qvel[self.qvel_rk]   # right knee slider velocity (m/s)
        
        
        # Contact information (6 additional dimensions)
        contact_info = self._compute_foot_contacts()
        ob[11] = contact_info['left_contact']      # Binary contact flag
        ob[12] = contact_info['right_contact']      # Binary contact flag
        ob[13] = contact_info['left_force']         # Normalized contact force
        ob[14] = contact_info['right_force']        # Normalized contact force
        ob[15] = contact_info['left_contact_duration']   # Time since contact switch
        ob[16] = contact_info['right_contact_duration']  # Time since contact switch
        
        return ob

    def _compute_foot_contacts(self) -> dict:
        """
        Compute foot contact information from MuJoCo contact data.
        
        Returns:
            Dictionary with contact flags, forces, and durations
        """
        dt = self.model.opt.timestep
        
        # Get contact forces for both feet
        left_force = self._get_contact_force(self.b_lfoot)
        right_force = self._get_contact_force(self.b_rfoot)
        
        # Determine contact state based on force threshold
        left_contact = left_force > self._contact_threshold
        right_contact = right_force > self._contact_threshold
        
        # Update contact durations
        if left_contact == self._prev_left_contact:
            self._left_contact_duration += dt
        else:
            self._left_contact_duration = 0.0
            self._prev_left_contact = left_contact
            
        if right_contact == self._prev_right_contact:
            self._right_contact_duration += dt
        else:
            self._right_contact_duration = 0.0
            self._prev_right_contact = right_contact
        
        return {
            'left_contact': float(left_contact),
            'right_contact': float(right_contact),
            'left_force': left_force / 100.0,  # Normalize to reasonable range
            'right_force': right_force / 100.0,  # Normalize to reasonable range
            'left_contact_duration': self._left_contact_duration,
            'right_contact_duration': self._right_contact_duration
        }
    
    def _get_contact_force(self, body_id: int) -> float:
        """
        Get contact force magnitude for a specific body.
        
        Args:
            body_id: MuJoCo body ID
            
        Returns:
            Contact force magnitude (N)
        """
        force_magnitude = 0.0
        
        # Get all geoms belonging to this body
        body_geoms = []
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                body_geoms.append(geom_id)
        
        # Iterate through all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # Check if this contact involves any geom of the specified body
            if contact.geom1 in body_geoms or contact.geom2 in body_geoms:
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, force)
                
                # Use normal force (z-component) as contact force
                normal_force = abs(force[2])
                force_magnitude += normal_force
        
        return force_magnitude

    def step(self, action: np.ndarray):
        """Advance simulation by one control step."""
        # Read current joint states
        qpos, qvel = self.data.qpos, self.data.qvel
        self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
        self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Update FSM (uses cached indices for performance)
        self.fsm.update(self.data, self.model)

        # Compute desired joint positions
        if self.mode == "fsm":
            # FSM mode: use FSM targets
            self._qdes[0] = self.fsm.desired_hip()
            lk_des, rk_des = self.fsm.desired_knees()
            self._qdes[1], self._qdes[2] = lk_des, rk_des
        elif self.mode == "hybrid_hip":
            # Hybrid mode: NN controls hip, FSM controls knees
            self._qdes[0] = self.pd.denorm(0, float(action[0]))  # NN hip
            lk_des, rk_des = self.fsm.desired_knees()  # FSM knees
            self._qdes[1], self._qdes[2] = lk_des, rk_des
        elif self.mode == "hybrid_knees":
            # Hybrid mode: FSM controls hip, NN controls knees
            self._qdes[0] = self.fsm.desired_hip()  # FSM hip
            self._qdes[1] = self.pd.denorm(1, float(action[1]))  # NN left knee
            self._qdes[2] = self.pd.denorm(2, float(action[2]))  # NN right knee
        else:
            # Research mode: use neural network actions for all joints
            self._qdes[0] = self.pd.denorm(0, float(action[0]))
            self._qdes[1] = self.pd.denorm(1, float(action[1]))
            self._qdes[2] = self.pd.denorm(2, float(action[2]))

        # Integrate micro-steps until target control time is reached
        n_microsteps = 0
        while self.data.time + 1e-12 < self._t_next:
            self._u[:] = self.pd.step(self._q, self._qd, self._qdes)
            self.data.ctrl[self.a_hip] = self._u[0]
            self.data.ctrl[self.a_lk] = self._u[1]
            self.data.ctrl[self.a_rk] = self._u[2]
            mujoco.mj_step(self.model, self.data)
            n_microsteps += 1

            # Update joint states for next micro-step
            qpos, qvel = self.data.qpos, self.data.qvel
            self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
            self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Timing debug logging (first few steps only)
        if self._timing_debug and self.data.time < 3.0:
            target_dt = 1.0 / self.ctrl_hz
            timestep = self.model.opt.timestep
            time_error = abs(self.data.time - self._t_next)
            if n_microsteps <= 5:  # Only log first few steps
                print(f"Timing: target_dt={target_dt:.4f}, timestep={timestep:.4f}, "
                      f"n_microsteps={n_microsteps}, time_error={time_error:.6f}")

        # Set next control step target
        self._t_next += 1.0 / self.ctrl_hz

        # Compute reward and termination
        current_x = float(self.data.qpos[self.qpos_x])
        dx = current_x - self.prev_x
        self.prev_x = current_x

        pitch_abs = abs(float(self.data.qpos[self.qpos_pitch]))
        ctrl_abs_sum = float(np.abs(self._u).sum())
        torso_z = float(self.data.xpos[self.b_torso, 2])
        
        # Additional signals for research mode
        velocity_x = float(self.data.qvel[self.dof_x])
        left_knee_pos = float(self.data.qpos[self.qpos_lk])
        right_knee_pos = float(self.data.qpos[self.qpos_rk])
        left_foot_z = float(self.data.xpos[self.b_lfoot, 2])
        right_foot_z = float(self.data.xpos[self.b_rfoot, 2])
        
        # Control change penalty (smooth motion)
        u_change_sum = float(np.abs(self._u - self._prev_u).sum()) if hasattr(self, '_prev_u') else 0.0
        self._prev_u = self._u.copy()

        signals = {
            "dx": dx,
            "pitch_abs": pitch_abs,
            "u_abs_sum": ctrl_abs_sum,
            "torso_z": torso_z,
            # Additional signals for research mode
            "velocity_x": velocity_x,
            "left_knee_pos": left_knee_pos,
            "right_knee_pos": right_knee_pos,
            "left_foot_z": left_foot_z,
            "right_foot_z": right_foot_z,
            "u_change_sum": u_change_sum,
        }

        reward, rinfo = self.reward_fn(signals)
        done = bool(rinfo["fell"] or (self.data.time >= self.simend))

        info = {
            "time": self.data.time,
            "dx": dx,
            "pitch_abs": pitch_abs,
            "torso_z": torso_z,
            "u_abs_sum": ctrl_abs_sum,
            "u": self._u.copy(),
            "qdes": self._qdes.copy(),
            "fsm_hip": self.fsm.fsm_hip,
            "fsm_k1": self.fsm.fsm_knee1,
            "fsm_k2": self.fsm.fsm_knee2,
        }
        # Add seed info on first step for dataset provenance
        if self._seed_used is not None and "seed" not in info:
            info["seed"] = self._seed_used
        info.update(rinfo)
        return self._get_obs(), float(reward), done, info

    def _ensure_window(self):
        """Initialize GUI window and rendering context."""
        if self.window is not None:
            return
        
        # Initialize GLFW (simple approach like legacy code)
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        
        # Create window (simple approach without OpenGL hints)
        self.window = glfw.create_window(1200, 900, "Passive Walker - v2.1", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed - check display server (X11/Wayland)")
        
        # Make context current
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        # Initialize MuJoCo rendering components (match legacy naming)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = float(CAM_DISTANCE)
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        print(f"✅ GUI window created: 1200x900")
        print(f"💡 Tip: If window not visible, try Alt+Tab")

    def render(self, mode: str = "human"):
        """Render the current state."""
        if not self.use_gui:
            return
        self._ensure_window()
        
        # Check if window is still valid
        if glfw.window_should_close(self.window):
            return
            
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam.lookat[0] = self.data.qpos[0]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """Clean up resources."""
        if self.use_gui and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None


# =====================
# Command Line Interface
# =====================
def main():
    """Command line interface for testing the environment."""
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser(description="Passive Walker Environment")
    parser.add_argument("--mode", type=str, default="fsm", choices=["fsm", "research", "hybrid_hip", "hybrid_knees"],
                        help="Control mode")
    parser.add_argument("--seconds", type=float, default=SIM_SECONDS,
                        help="Episode length (s)")
    parser.add_argument("--gui", dest="gui", action="store_true",
                        help="Enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Disable GUI")
    parser.set_defaults(gui=DEFAULT_GUI)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    # Physics randomization arguments
    parser.add_argument("--ramp-jitter", type=float, default=0.0, help="Ramp angle variation (degrees)")
    parser.add_argument("--friction-min", type=float, default=0.6, help="Minimum friction coefficient")
    parser.add_argument("--friction-max", type=float, default=1.0, help="Maximum friction coefficient")
    parser.add_argument("--randomization-profile", type=str, choices=["none", "basic", "moderate", "aggressive", "temporal"],
                        help="Advanced randomization profile")
    parser.add_argument("--ctrl-hz", type=float, default=100.0, help="Control frequency (Hz)")
    
    # PD Backend selection (mutually exclusive)
    # Default: NumPy (fastest for single environment)
    # JAX: Better for vectorized/batched operations
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--jax-pd", action="store_true",
                              help="use JAX PD backend (JIT-compiled, better for vectorized operations)")
    backend_group.add_argument("--numpy-pd", action="store_true",
                              help="use NumPy PD backend (default, fastest for single environment)")
    args = parser.parse_args()

    # Determine backend: Check environment variable first, then CLI flags
    import os
    env_default = os.environ.get("PWALKER_PD_BACKEND", "numpy").lower()
    env_wants_jax = (env_default == "jax")
    use_jax_pd = (args.jax_pd or (env_wants_jax and not args.numpy_pd))
    # Try GUI first, fallback to headless if it fails
    use_gui = args.gui
    if use_gui:
        try:
            env = PassiveWalkerEnv(
                mode=args.mode, 
                use_gui=True, 
                use_jax_pd=use_jax_pd,
                ramp_jitter=args.ramp_jitter,
                friction_min=args.friction_min,
                friction_max=args.friction_max,
                randomization_profile=args.randomization_profile,
                ctrl_hz=args.ctrl_hz
            )
            print("✅ GUI mode enabled")
        except Exception as e:
            print(f"⚠️  GUI failed: {e}")
            print("🔄 Falling back to headless mode...")
            use_gui = False
            env = PassiveWalkerEnv(
                mode=args.mode, 
                use_gui=False, 
                use_jax_pd=use_jax_pd,
                ramp_jitter=args.ramp_jitter,
                friction_min=args.friction_min,
                friction_max=args.friction_max,
                randomization_profile=args.randomization_profile,
                ctrl_hz=args.ctrl_hz
            )
    else:
        env = PassiveWalkerEnv(
            mode=args.mode, 
            use_gui=False, 
            use_jax_pd=use_jax_pd,
            ramp_jitter=args.ramp_jitter,
            friction_min=args.friction_min,
            friction_max=args.friction_max,
            randomization_profile=args.randomization_profile,
            ctrl_hz=args.ctrl_hz
        )
    env.simend = float(args.seconds)

    obs, _ = env.reset(seed=args.seed)
    zero = np.zeros(_ACT_DIM, dtype=np.float32)

    t0 = time.time()
    last = t0
    steps = 0
    rsum = 0.0

    try:
        done = False
        while not done:
            if use_gui and env.window is not None and glfw.window_should_close(env.window):
                break
            obs, r, done, info = env.step(zero)
            rsum += r
            steps += 1

            now = time.time()
            if now - last >= 0.5:
                fps = steps / (now - last)
                print(f"time={info.get('time', 0):6.3f}  dx={info.get('dx', 0):+.4f}  "
                      f"pitch|rad|={info.get('pitch_abs', 0):.3f}  "
                      f"torso_z={info.get('torso_z', 0):.3f}  "
                      f"r_step={r:+.4f}  r_sum={rsum:+.2f}  fell={info.get('fell', False)}  "
                      f"fps~{fps:5.1f}")
                last = now
                steps = 0
            env.render()
    finally:
        # Print backend once at end
        print(f"[PassiveWalker] PD backend: {env.pd.backend_name}")
        env.close()


if __name__ == "__main__":
    main()