"""
Unified Passive Walker Environment

A single environment class supporting both FSM data collection and RL research modes.
Uses YAML configuration, PD control, and configurable reward functions.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import gym
from gym import spaces
import mujoco
from mujoco.glfw import glfw

from .config import WalkerConfig
from .controller import PDController, FSMStateMachine
from .reward import get_reward_fn

# Environment dimensions
_OBS_DIM = 11  # [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
_ACT_DIM = 3  # [hip, left_knee, right_knee]


class PassiveWalkerEnv(gym.Env):
    """
    Unified bipedal walker environment supporting two modes:
    - 'fsm': Data collection mode with minimal rewards
    - 'research': RL training mode with shaped rewards
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, cfg: WalkerConfig, use_gui: bool = True):
        """Initialize environment with configuration."""
        super().__init__()
        self.cfg = cfg
        self.use_gui = use_gui
        self.ctrl_hz = cfg.env.ctrl_hz
        self.simend = cfg.env.simend

        # Initialize MuJoCo model and data
        self.window = None
        self._init_model_and_ids(cfg.env.xml_path)

        # Define observation and action spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACT_DIM,), dtype=np.float32)

        # Initialize controllers and reward function
        self.pd = PDController(cfg.control)
        self.fsm = FSMStateMachine()

        # Set reward function based on mode
        if cfg.mode == "fsm":
            self.reward_fn = get_reward_fn("minimal")
        else:  # research mode
            self.reward_fn = get_reward_fn(cfg.reward.preset, cfg.reward.overrides)

        # Pre-allocated arrays for performance
        self._obs = np.empty(_OBS_DIM, dtype=np.float32)
        self._q = np.empty(3, dtype=np.float32)
        self._qd = np.empty(3, dtype=np.float32)
        self._qdes = np.zeros(3, dtype=np.float32)
        self._u = np.empty(3, dtype=np.float32)
        self.prev_x = 0.0

        # Initialize GUI if requested
        if self.use_gui:
            self._ensure_window()

    def _init_model_and_ids(self, xml_path: str) -> None:
        """Load MuJoCo model and cache body/joint IDs for performance."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache joint/body/actuator IDs for fast access
        def jid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        def bid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        def aid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        # Joint IDs and addresses
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
        self.b_lleg = bid("left_leg")
        self.b_rleg = bid("right_leg")

        # Actuator IDs
        self.a_hip = aid("hip_act")
        self.a_lk = aid("left_knee_act")
        self.a_rk = aid("right_knee_act")

        # Set initial ramp tilt
        tilt = np.deg2rad(self.cfg.physics.ramp_deg_min)
        self.model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state."""
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Initialize FSM and set nominal pose
        self.fsm.reset()
        self.data.qpos[self.qpos_hip] = -0.5
        self.data.qpos[self.qpos_lk] = 0.0
        self.data.qpos[self.qpos_rk] = 0.0
        self.data.ctrl[self.a_hip] = self.data.ctrl[self.a_lk] = self.data.ctrl[self.a_rk] = 0.0

        # Forward kinematics and reset state
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self.prev_x = float(self.data.qpos[self.dof_x])

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Extract observation vector: [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]."""
        ob = self._obs
        qpos, qvel = self.data.qpos, self.data.qvel

        # Position and velocity
        ob[0] = qpos[self.dof_x]  # x position
        ob[1] = qpos[self.dof_z]  # z position
        ob[2] = qpos[self.qpos_pitch]  # pitch angle
        ob[3] = qvel[self.dof_x]  # x velocity
        ob[4] = qvel[self.dof_z]  # z velocity

        # Joint positions and velocities
        ob[5] = qpos[self.qpos_hip]  # hip angle
        ob[6] = qpos[self.qpos_lk]  # left knee position
        ob[7] = qpos[self.qpos_rk]  # right knee position
        ob[8] = qvel[self.qvel_hip]  # hip velocity
        ob[9] = qvel[self.qvel_lk]  # left knee velocity
        ob[10] = qvel[self.qvel_rk]  # right knee velocity

        return ob

    def step(self, action: np.ndarray):
        """Execute one environment step with action."""
        # Get current joint states
        qpos, qvel = self.data.qpos, self.data.qvel
        self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
        self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Update FSM state
        self.fsm.update(self.data, self.model)

        # Determine desired joint positions (NN vs FSM)
        use_nn_hip = self.cfg.control.use_nn_for_hip
        use_nn_knees = self.cfg.control.use_nn_for_knees

        self._qdes[:] = 0.0
        if use_nn_hip:
            self._qdes[0] = self.pd.denorm(0, action[0])
        else:
            self._qdes[0] = self.fsm.desired_hip()

        if use_nn_knees:
            self._qdes[1] = self.pd.denorm(1, action[1])
            self._qdes[2] = self.pd.denorm(2, action[2])
        else:
            lk_des, rk_des = self.fsm.desired_knees()
            self._qdes[1], self._qdes[2] = lk_des, rk_des

        # Apply PD control and simulate physics
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

        # Compute reward and check termination
        current_x = float(self.data.qpos[self.dof_x])
        dx = current_x - self.prev_x
        self.prev_x = current_x

        # Extract state signals for reward function
        pitch_abs = abs(float(self.data.qpos[self.qpos_pitch]))
        ctrl_abs_sum = float(np.sum(np.abs(self._u)))
        left_z = float(self.data.xpos[self.b_lfoot, 2])
        right_z = float(self.data.xpos[self.b_rfoot, 2])
        torso_z = float(self.data.xpos[self.b_torso, 2])
        vx = float(self.data.qvel[self.dof_x])

        # Build signals dictionary for reward computation
        signals = {
            "dx": dx,
            "pitch_abs": pitch_abs,
            "u_abs_sum": ctrl_abs_sum,
            "torso_z": torso_z,
            "vx": vx,
            "lk_q": float(self.data.qpos[self.qpos_lk]),
            "rk_q": float(self.data.qpos[self.qpos_rk]),
            "left_foot_z": left_z,
            "right_foot_z": right_z,
        }

        # Compute reward and check for fall
        reward, rinfo = self.reward_fn(signals)
        fell = rinfo["fell"]
        done = fell or (self.data.time >= self.simend)

        # Build info dictionary
        info = {
            "time": self.data.time,
            "dx": dx,
            "pitch_abs": pitch_abs,
            "torso_z": torso_z,
            "vx": vx,
        }
        info.update(rinfo)  # Add reward breakdown

        return self._get_obs(), float(reward), bool(done), info

    def _ensure_window(self):
        """Initialize GLFW window and MuJoCo rendering context."""
        if self.window is not None:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1200, 900, "Passive Walker", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 8.0
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    def render(self, mode="human"):
        """Render the environment."""
        if not self.use_gui:
            return
        self._ensure_window()
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam.lookat[0] = self.data.qpos[0]
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mujoco.mjr_render(viewport, self.scene, self.ctx)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        """Clean up resources."""
        if self.use_gui and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None


if __name__ == "__main__":
    # Quick FSM demo runner for manual inspection / debugging.
    import argparse
    import time
    import numpy as np

    from .io import load_walker_config

    parser = argparse.ArgumentParser(description="PassiveWalkerEnv demo runner")
    parser.add_argument(
        "--config",
        type=str,
        default="passive_walker/configs/walker.yaml",
        help="Path to walker YAML config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fsm",
        choices=["fsm", "research"],
        help="Environment mode",
    )
    parser.add_argument(
        "--seconds", type=float, default=20.0, help="Sim time limit (s)"
    )
    parser.add_argument(
        "--gui", dest="gui", action="store_true", help="Enable GUI (default)"
    )
    parser.add_argument(
        "--no-gui", dest="gui", action="store_false", help="Disable GUI"
    )
    parser.set_defaults(gui=True)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reset()")
    # Optional: let NN override FSM for quick checks
    parser.add_argument("--nn-hip", action="store_true", help="Use NN action for hip")
    parser.add_argument(
        "--nn-knees", action="store_true", help="Use NN actions for knees"
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Telemetry print interval (s)",
    )

    args = parser.parse_args()

    # Load and tweak config
    cfg = load_walker_config(args.config)
    cfg.mode = args.mode
    cfg.env.simend = float(args.seconds)
    cfg.control.use_nn_for_hip = bool(args.nn_hip)
    cfg.control.use_nn_for_knees = bool(args.nn_knees)

    env = PassiveWalkerEnv(cfg, use_gui=args.gui)
    obs, _ = env.reset(seed=args.seed)

    zero_act = np.zeros(3, dtype=np.float32)  # FSM ignores unless --nn-hip/--nn-knees

    t0 = time.time()
    last = t0
    steps = 0
    total_r = 0.0
    try:
        done = False
        while not done:
            if args.gui and env.window is not None:
                from mujoco.glfw import glfw

                if glfw.window_should_close(env.window):
                    break

            obs, r, done, info = env.step(zero_act)
            total_r += r
            steps += 1

            now = time.time()
            if (now - last) >= args.print_interval:
                fps = steps / (now - last)
                print(
                    f"time={info.get('time', 0):6.3f}  "
                    f"dx={info.get('dx', 0):+.4f}  vx={info.get('vx', 0):+.3f}  "
                    f"pitch|rad|={info.get('pitch_abs', 0):.3f}  "
                    f"torso_z={info.get('torso_z', 0):.3f}  "
                    f"r_step={r:+.4f}  r_sum={total_r:+.2f}  "
                    f"fell={info.get('fell', False)}  fps~{fps:5.1f}"
                )
                last = now
                steps = 0

            env.render()

    finally:
        env.close()
