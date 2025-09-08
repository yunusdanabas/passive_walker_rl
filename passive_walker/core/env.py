"""
Passive Walker Environment

Bipedal walking environment with FSM and neural network control modes.
Supports both headless and GUI operation with proper physics simulation.
"""
from __future__ import annotations
from typing import Optional

# =====================
# Configuration
# =====================
# Simulation parameters
CTRL_HZ = 50.0          # Controller frequency (Hz)
SIM_SECONDS = 20.0      # Default episode length (s)
XML_PATH = "passive_walker/assets/passiveWalker_model.xml"

# Physics parameters
RAMP_DEG = 10.0         # Incline angle (degrees, positive = downhill)
FRICTION = 0.9          # Contact friction coefficient
RANDOMIZE_PHYSICS = False  # Enable domain randomization
MASS_JITTER = 0.05      # ±5% torso mass variation when randomization enabled

# Rendering parameters
DEFAULT_GUI = True      # Enable GUI when run as module
CAM_DISTANCE = 8.0      # Camera distance from walker

# Observation and action dimensions
_OBS_DIM = 11  # [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
_ACT_DIM = 3   # [hip, left_knee, right_knee]


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from mujoco.glfw import glfw

from .controller import PDController, FSMStateMachine
from .reward import get_reward_fn


class PassiveWalkerEnv(gym.Env):
    """
    Passive Walker Environment with FSM and Neural Network control.
    
    Modes:
        - 'fsm': Uses built-in FSM for stable walking (actions ignored)
        - 'research': Uses neural network actions for learning
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, mode: str = "fsm", use_gui: bool = False):
        super().__init__()
        assert mode in ("fsm", "research")
        self.mode = mode
        self.use_gui = use_gui
        self.ctrl_hz = float(CTRL_HZ)
        self.simend = float(SIM_SECONDS)

        self.window = None
        self._np_rng = None  # Random number generator

        # Initialize MuJoCo model and body/joint IDs
        self._init_model_and_ids(XML_PATH)

        # Define observation and action spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACT_DIM,), dtype=np.float32)

        # Initialize controllers
        self.pd = PDController()
        self.fsm = FSMStateMachine()

        # Initialize reward function
        self.reward_fn = get_reward_fn(self.mode)

        # Pre-allocate arrays for performance
        self._obs = np.empty(_OBS_DIM, dtype=np.float32)
        self._q = np.empty(3, dtype=np.float32)
        self._qd = np.empty(3, dtype=np.float32)
        self._qdes = np.zeros(3, dtype=np.float32)
        self._u = np.empty(3, dtype=np.float32)

        self.prev_x = 0.0

        if self.use_gui:
            self._ensure_window()

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
        self.j_slide_z = jid("slide_z")
        self.dof_z = self.model.jnt_dofadr[self.j_slide_z]
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

    def _apply_base_physics(self):
        """Set up base physics parameters (gravity, friction)."""
        tilt = np.deg2rad(RAMP_DEG)
        self.model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]
        self.model.geom_friction[:, 0] = float(FRICTION)

    def _randomize_physics(self):
        """Apply domain randomization to physics parameters."""
        assert self._np_rng is not None
        # Apply base physics
        self._apply_base_physics()
        # Randomize torso mass
        base_mass = float(self.model.body_mass[self.b_torso])
        scale = self._np_rng.uniform(1.0 - MASS_JITTER, 1.0 + MASS_JITTER)
        self.model.body_mass[self.b_torso] = base_mass * scale

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None or self._np_rng is None:
            self._np_rng = np.random.RandomState(None if seed is None else int(seed))

        mujoco.mj_resetData(self.model, self.data)
        self.fsm.reset()

        # Set initial pose (hip at -0.5 to match legacy FSM)
        self.data.qpos[self.qpos_hip] = -0.5
        self.data.qpos[self.qpos_lk] = 0.0
        self.data.qpos[self.qpos_rk] = 0.0
        self.data.ctrl[self.a_hip] = self.data.ctrl[self.a_lk] = self.data.ctrl[self.a_rk] = 0.0

        # Apply physics (base or randomized)
        if RANDOMIZE_PHYSICS:
            self._randomize_physics()
        else:
            self._apply_base_physics()

        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self.prev_x = float(self.data.qpos[self.dof_x])

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation vector."""
        qpos, qvel = self.data.qpos, self.data.qvel
        ob = self._obs
        ob[0] = qpos[self.dof_x]      # x position
        ob[1] = qpos[self.dof_z]      # z position
        ob[2] = qpos[self.qpos_pitch] # pitch angle
        ob[3] = qvel[self.dof_x]      # x velocity
        ob[4] = qvel[self.dof_z]      # z velocity
        ob[5] = qpos[self.qpos_hip]   # hip angle
        ob[6] = qpos[self.qpos_lk]    # left knee position
        ob[7] = qpos[self.qpos_rk]    # right knee position
        ob[8] = qvel[self.qvel_hip]   # hip angular velocity
        ob[9] = qvel[self.qvel_lk]    # left knee velocity
        ob[10] = qvel[self.qvel_rk]   # right knee velocity
        return ob

    def step(self, action: np.ndarray):
        """Advance simulation by one control step."""
        # Read current joint states
        qpos, qvel = self.data.qpos, self.data.qvel
        self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
        self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Update FSM with leg body IDs for accurate angle detection
        self.fsm.update(self.data, self.model, self.b_lfoot, self.b_rfoot,
                        self.b_lleg, self.b_rleg)

        # Compute desired joint positions
        if self.mode == "fsm":
            # FSM mode: use FSM targets
            self._qdes[0] = self.fsm.desired_hip()
            lk_des, rk_des = self.fsm.desired_knees()
            self._qdes[1], self._qdes[2] = lk_des, rk_des
        else:
            # Research mode: use neural network actions
            self._qdes[0] = self.pd.denorm(0, float(action[0]))
            self._qdes[1] = self.pd.denorm(1, float(action[1]))
            self._qdes[2] = self.pd.denorm(2, float(action[2]))

        # Integrate multiple micro-steps to match control frequency
        sim_steps = max(1, int((1.0 / self.ctrl_hz) / self.model.opt.timestep))
        for _ in range(sim_steps):
            self._u[:] = self.pd.step(self._q, self._qd, self._qdes)
            self.data.ctrl[self.a_hip] = self._u[0]
            self.data.ctrl[self.a_lk] = self._u[1]
            self.data.ctrl[self.a_rk] = self._u[2]
            mujoco.mj_step(self.model, self.data)

            # Update joint states for next micro-step
            qpos, qvel = self.data.qpos, self.data.qvel
            self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
            self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Compute reward and termination
        current_x = float(self.data.qpos[self.dof_x])
        dx = current_x - self.prev_x
        self.prev_x = current_x

        pitch_abs = abs(float(self.data.qpos[self.qpos_pitch]))
        ctrl_abs_sum = float(np.abs(self._u).sum())
        torso_z = float(self.data.xpos[self.b_torso, 2])

        signals = {
            "dx": dx,
            "pitch_abs": pitch_abs,
            "u_abs_sum": ctrl_abs_sum,
            "torso_z": torso_z,
        }

        reward, rinfo = self.reward_fn(signals)
        done = bool(rinfo["fell"] or (self.data.time >= self.simend))

        info = {
            "time": self.data.time,
            "dx": dx,
            "pitch_abs": pitch_abs,
            "torso_z": torso_z,
        }
        info.update(rinfo)
        return self._get_obs(), float(reward), done, info

    def _ensure_window(self):
        """Initialize GUI window and rendering context."""
        if self.window is not None:
            return
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        self.window = glfw.create_window(1200, 900, "Passive Walker", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = float(CAM_DISTANCE)
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    def render(self, mode: str = "human"):
        """Render the current state."""
        if not self.use_gui:
            return
        self._ensure_window()
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam.lookat[0] = self.data.qpos[0]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.ctx)
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
    parser.add_argument("--mode", type=str, default="fsm", choices=["fsm", "research"],
                        help="Control mode")
    parser.add_argument("--seconds", type=float, default=SIM_SECONDS,
                        help="Episode length (s)")
    parser.add_argument("--gui", dest="gui", action="store_true",
                        help="Enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Disable GUI")
    parser.set_defaults(gui=DEFAULT_GUI)
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    env = PassiveWalkerEnv(mode=args.mode, use_gui=args.gui)
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
            if args.gui and env.window is not None and glfw.window_should_close(env.window):
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
        env.close()


if __name__ == "__main__":
    main()