"""
Unified PassiveWalkerEnv.

- Single class with two modes, toggled via config.mode: "fsm" or "research".
- Reads dataclasses from YAML.
- Uses controller.PDController (FSM logic pluggable).
- Uses reward.compute_reward(...) with chosen preset.
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

# ---- Constants kept small and local -------------------------------------------------
_OBS_DIM = 11  # [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
_ACT_DIM = 3  # [hip, left_knee, right_knee]


class PassiveWalkerEnv(gym.Env):
    """
    Unified bipedal environment with two modes:
      - 'fsm': minimal reward, rich logging; used for expert data collection
      - 'research': shaped reward presets for RL
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, cfg: WalkerConfig, use_gui: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_gui = use_gui
        self.ctrl_hz = cfg.env.ctrl_hz
        self.simend = cfg.env.simend

        # ---- Window is lazy; model/data are not
        self.window = None
        self._init_model_and_ids(cfg.env.xml_path)

        # ---- Spaces (always the same)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(_ACT_DIM,), dtype=np.float32)

        # ---- Controller + FSM
        self.pd = PDController(cfg.control)
        self.fsm = FSMStateMachine(cfg.fsm)  # lightweight: only stores state; uses sim data to transition

        # ---- Reward (preset chosen in YAML: minimal/default/aggressive)
        if cfg.mode == "fsm":
            self.reward_fn = get_reward_fn("minimal")
        else:  # research mode
            self.reward_fn = get_reward_fn(cfg.reward.preset, cfg.reward.overrides)

        # ---- Random number generator for deterministic DR
        self._np_rng = np.random.RandomState(42)  # Default seed

        # ---- Preallocated scratch (avoid per-step allocs)
        self._obs = np.empty(_OBS_DIM, dtype=np.float32)
        self._q = np.empty(3, dtype=np.float32)
        self._qd = np.empty(3, dtype=np.float32)
        self._qdes = np.zeros(3, dtype=np.float32)
        self._u = np.empty(3, dtype=np.float32)

        # ---- Progress helpers
        self.prev_x = 0.0

        # ---- Optional GUI setup
        if self.use_gui:
            self._ensure_window()

    # -------------------------- MuJoCo setup -----------------------------------------
    def _init_model_and_ids(self, xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Cache joint ids/addrs (fast indexing)
        def jid(n):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)

        def bid(n):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

        def aid(n):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

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

        self.b_torso = bid("torso")
        self.b_lfoot = bid("left_foot")
        self.b_rfoot = bid("right_foot")
        self.b_lleg = bid("left_leg")
        self.b_rleg = bid("right_leg")

        self.a_hip = aid("hip_act")
        self.a_lk = aid("left_knee_act")
        self.a_rk = aid("right_knee_act")

        # Initial ramp tilt (overridable by domain randomization later)
        tilt = np.deg2rad(self.cfg.physics.ramp_deg_min)
        self.model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]

    def _randomize_physics(self) -> None:
        """Apply domain randomization to physics parameters."""
        # Ramp tilt
        tilt_deg = self._np_rng.uniform(
            self.cfg.physics.ramp_deg_min,
            self.cfg.physics.ramp_deg_max
        )
        tilt = np.deg2rad(tilt_deg)
        self.model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]

        # Friction (geom 0: slide friction column)
        mu = self._np_rng.uniform(*self.cfg.physics.friction)
        self.model.geom_friction[:, 0] = mu

        # Torso mass jitter
        torso = self.b_torso
        base_mass = float(self.model.body_mass[torso])
        scale = self._np_rng.uniform(
            1.0 - self.cfg.physics.mass_jitter,
            1.0 + self.cfg.physics.mass_jitter
        )
        self.model.body_mass[torso] = base_mass * scale

    # -------------------------- Reset/Obs --------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)
        mujoco.mj_resetData(self.model, self.data)

        # Domain randomization
        if self.cfg.physics.randomize_physics:
            self._randomize_physics()

        # FSM boot
        self.fsm.reset()

        # Nominal pose
        self.data.qpos[self.qpos_hip] = -0.5
        self.data.qpos[self.qpos_lk] = 0.0
        self.data.qpos[self.qpos_rk] = 0.0
        self.data.ctrl[self.a_hip] = self.data.ctrl[self.a_lk] = self.data.ctrl[self.a_rk] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self.prev_x = float(self.data.qpos[self.dof_x])

        return self._get_obs(), {}

    def seed(self, seed: int):
        """Set the random seed for deterministic domain randomization."""
        self._np_rng = np.random.RandomState(seed)
        return [seed]

    def _get_obs(self) -> np.ndarray:
        ob = self._obs
        qpos, qvel = self.data.qpos, self.data.qvel
        ob[0] = qpos[self.dof_x]
        ob[1] = qpos[self.dof_z]
        ob[2] = qpos[self.qpos_pitch]
        ob[3] = qvel[self.dof_x]
        ob[4] = qvel[self.dof_z]
        ob[5] = qpos[self.qpos_hip]
        ob[6] = qpos[self.qpos_lk]
        ob[7] = qpos[self.qpos_rk]
        ob[8] = qvel[self.qvel_hip]
        ob[9] = qvel[self.qvel_lk]
        ob[10] = qvel[self.qvel_rk]
        return ob

    # -------------------------- Step -----------------------------------------------
    def step(self, action: np.ndarray):
        # Convert normalized action to desired positions if NN is enabled in config.
        # Otherwise, desired positions come from FSM.
        use_nn_hip = self.cfg.control.use_nn_for_hip
        use_nn_knees = self.cfg.control.use_nn_for_knees

        # Pre-fill q, qd
        qpos, qvel = self.data.qpos, self.data.qvel
        self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
        self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # Update FSM states (FSM is always updated; it's read if that limb isn't NN-controlled)
        self.fsm.update(self.data, self.model)

        # Desired joint positions
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

        # PD control and integrate physics for the control slice
        sim_steps = max(1, int((1.0 / self.ctrl_hz) / self.model.opt.timestep))
        for _ in range(sim_steps):
            self._u[:] = self.pd.step(self._q, self._qd, self._qdes)
            self.data.ctrl[self.a_hip] = self._u[0]
            self.data.ctrl[self.a_lk] = self._u[1]
            self.data.ctrl[self.a_rk] = self._u[2]
            mujoco.mj_step(self.model, self.data)

            # refresh q, qd for next micro-step without realloc
            qpos, qvel = self.data.qpos, self.data.qvel
            self._q[:] = [qpos[self.qpos_hip], qpos[self.qpos_lk], qpos[self.qpos_rk]]
            self._qd[:] = [qvel[self.qvel_hip], qvel[self.qvel_lk], qvel[self.qvel_rk]]

        # ----- Reward & termination
        current_x = float(self.data.qpos[self.dof_x])
        dx = current_x - self.prev_x
        self.prev_x = current_x

        # Shared signals
        pitch_abs = abs(float(self.data.qpos[self.qpos_pitch]))
        ctrl_abs_sum = float(np.sum(np.abs(self._u)))
        left_z = float(self.data.xpos[self.b_lfoot, 2])
        right_z = float(self.data.xpos[self.b_rfoot, 2])
        torso_z = float(self.data.xpos[self.b_torso, 2])
        vx = float(self.data.qvel[self.dof_x])

        # Build signals dict for reward function
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

        # Compute reward and extract fell status
        reward, rinfo = self.reward_fn(signals)
        fell = rinfo["fell"]

        # Additional termination conditions
        fell = fell or (pitch_abs > self.cfg.terminations.fall_pitch_max) or (torso_z < self.cfg.terminations.fall_z_min)
        stalled = abs(vx) < self.cfg.terminations.max_idle_speed
        unstable = pitch_abs > 0.5

        done = fell or (self.data.time >= self.simend)
        if self.cfg.terminations.enable_stall_termination:
            done = done or stalled

        # Rich info dictionary
        info = {
            "time": self.data.time,
            "dx": dx,
            "pitch_abs": pitch_abs,
            "torso_z": torso_z,
            "vx": vx,
            "fell": fell,
            "stalled": stalled,
            "unstable": unstable,
        }
        
        # Quality metrics (if debug logging enabled)
        if self.cfg.debug.log_quality:
            stability = max(0.0, 1.0 - pitch_abs / 0.5)
            motion = 1.0 if 0.1 <= abs(vx) <= 2.0 else 0.5
            clearance = 1.0 if min(left_z, right_z) > 0.02 else 0.0
            info["quality_score"] = stability + motion + clearance
            
        # FSM state logging (if debug enabled)
        if self.cfg.debug.log_fsm:
            info["fsm_state"] = self.fsm.hip_state
            info["knee_states"] = self.fsm.knee_states.copy()
            
        info.update(rinfo)  # Add reward breakdown
        return self._get_obs(), float(reward), bool(done), info

    # -------------------------- Render / Close --------------------------------------
    def _ensure_window(self):
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
        if not self.use_gui:
            return None
        self._ensure_window()
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        self.cam.lookat[0] = self.data.qpos[0]
        self.cam.distance = self.cfg.render.camera_distance
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
        
        if mode == "rgb_array":
            img_w = self.cfg.render.rgb_array_width
            img_h = self.cfg.render.rgb_array_height
            px = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(px, None, mujoco.MjrRect(0, 0, img_w, img_h), self.ctx)
            glfw.poll_events()
            return np.flipud(px)  # MuJoCo origin at bottom-left
        else:
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            return None

    def close(self):
        if self.use_gui and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
