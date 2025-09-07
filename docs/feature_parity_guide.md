# Legacy-Feature Parity Guide for `PassiveWalkerEnv`

## 1) Config surface first (no magic numbers)

**Goal:** every threshold/toggle lives in YAML → dataclasses → env.

Extend `WalkerConfig` with:

* `physics`: `ramp_deg_min`, `ramp_deg_max`, `friction: [min, max]`, `mass_jitter`, `randomize_physics: bool`
* `termination`: `fall_pitch_max`, `fall_z_min`, `max_idle_speed`, `enable_stall_termination`
* `fsm`: `contact_height`, `knee_release_threshold`, `hip_swing_pos`, `hip_swing_neg`, `knee_stance`, `knee_retract`
* `render`: `camera_distance`, `rgb_array_width`, `rgb_array_height`
* `debug`: `log_quality: bool`, `log_fsm: bool`

**Done when:** env reads only `cfg.*` (no hard-coded floats).

---

## 2) Physics domain randomization (DR)

**Add to env (once):**

```python
def _randomize_physics(self) -> None:
    # Ramp tilt
    tilt_deg = np.random.uniform(self.cfg.physics.ramp_deg_min,
                                 self.cfg.physics.ramp_deg_max)
    tilt = np.deg2rad(tilt_deg)
    self.model.opt.gravity[:] = [9.81*np.sin(tilt), 0.0, -9.81*np.cos(tilt)]

    # Friction (geom 0: slide column)
    mu = np.random.uniform(*self.cfg.physics.friction)
    self.model.geom_friction[:, 0] = mu

    # Torso mass jitter
    torso = self.b_torso
    base = float(self.model.body_mass[torso])
    scale = np.random.uniform(1.0 - self.cfg.physics.mass_jitter,
                              1.0 + self.cfg.physics.mass_jitter)
    self.model.body_mass[torso] = base * scale
```

**In `reset()`:**

```python
if getattr(self.cfg.physics, "randomize_physics", False):
    self._randomize_physics()
```

**Done when:** `randomize_physics: true` changes gravity/friction/mass per episode.

---

## 3) FSM parameterization parity

Move legacy thresholds/targets into YAML and drive your `FSMStateMachine` from config:

* YAML: `fsm.contact_height`, `fsm.knee_release_threshold`, `fsm.hip_swing_pos/neg`, `fsm.knee_stance/retract`
* Update `FSMStateMachine.update()/desired_*()` to use `cfg.fsm.*`.

**Done when:** changing YAML shifts gait transitions.

---

## 4) Termination parity (fall + optional stall)

**After reward in `step()`:**

```python
fell = (pitch_abs > self.cfg.termination.fall_pitch_max) \
    or (torso_z < self.cfg.termination.fall_z_min)

stalled = abs(vx) < self.cfg.termination.max_idle_speed
done = fell or (self.data.time >= self.simend)
if self.cfg.termination.enable_stall_termination:
    done = done or stalled
```

Include flags in `info` (see §8).

---

## 5) Action denorm & PD parity

Ensure **all** control params are in config and applied via `PDController`:

* Gains/limits in `cfg.control` (arrays)
* Per-joint ranges used in `PDController.denorm(j, a)`
* Torque clipping inside `PDController.step(...)`

**Done when:** hot loop has zero hard-coded PD numbers.

---

## 6) Reward signals (optional parity knobs)

You already pass a compact `signals` dict. To match legacy shaping when needed, support opt-in keys (computed lazily only if preset asks):

* `prev_pitch_abs`, `foot_clearance_pair_min`, `knee_diff_abs`

**Done when:** legacy shaping is expressible as presets/overrides without touching env logic.

---

## 7) RGB array rendering

Implement the advertised mode:

```python
def render(self, mode="human"):
    if not self.use_gui:
        return None
    self._ensure_window()
    w, h = glfw.get_framebuffer_size(self.window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    self.cam.lookat[0] = self.data.qpos[0]
    mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                           mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
    mujoco.mjr_render(viewport, self.scene, self.ctx)
    if mode == "rgb_array":
        img_w = getattr(self.cfg.render, "rgb_array_width", w)
        img_h = getattr(self.cfg.render, "rgb_array_height", h)
        px = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(px, None, mujoco.MjrRect(0, 0, img_w, img_h), self.ctx)
        glfw.poll_events()
        return np.flipud(px)
    glfw.swap_buffers(self.window)
    glfw.poll_events()
```

---

## 8) Rich info & quality metrics (BC filtering)

Add parity flags & optional quality score:

```python
info.update({
    "fell": fell,
    "stalled": stalled,
    "unstable": pitch_abs > 0.5,
})

if getattr(self.cfg.debug, "log_quality", False):
    left_z = float(self.data.xpos[self.b_lfoot, 2])
    right_z = float(self.data.xpos[self.b_rfoot, 2])
    stability = max(0.0, 1.0 - pitch_abs/0.5)
    motion = 1.0 if 0.1 <= abs(vx) <= 2.0 else 0.5
    clearance = 1.0 if min(left_z, right_z) > 0.02 else 0.0
    info["quality_score"] = stability + motion + clearance

if getattr(self.cfg.debug, "log_fsm", False):
    info["fsm_state"] = self.fsm.state_id
```

---

## 9) Seeding & determinism (Gym 0.21)

* Keep `self._np_rng = np.random.RandomState(seed)` and use it in `_randomize_physics()` for deterministic DR when `seed` provided.
* Continue returning `(obs, info)` from `reset()`.

---

## 10) Tests & acceptance (fast)

Add/extend pytest:

* DR toggles: gravity/friction/mass change only when enabled.
* Termination: force pitch/height to trigger fall; toggle stall termination.
* FSM thresholds: adjust `contact_height` and confirm desired set-points change.
* RGB array: `render('rgb_array')` returns `(H, W, 3) uint8`.
* Seeding: DR reports repeat across runs with same seed.

---

## 11) Performance hygiene

* Preallocate arrays (already done).
* Compute optional signals only if preset requests (reward function can expose `required_keys`).
* Avoid building large dicts per step; reuse scalar locals.

---

## 12) Minimal touchpoints

* `WalkerConfig`: add `physics`, `termination`, `fsm`, `render`, `debug`
* `__init__`: strip magic numbers; default gravity from `ramp_deg_min`
* `reset()`: optional `_randomize_physics()`, reset FSM, set `prev_x`
* `step()`: add stall flag + richer `info`; keep reward call
* `render()`: implement `rgb_array`

---

## Quick checklist

* [ ] YAML exposes DR + FSM thresholds + terminations + render sizes
* [ ] `_randomize_physics()` integrated and deterministic under seed
* [ ] Stall termination toggle implemented
* [ ] Rich info metrics: `fell`, `stalled`, `unstable`, `quality_score`, `fsm_state`
* [ ] `rgb_array` implemented and returns `uint8`
* [ ] PD gains/limits/joint ranges all from config
* [ ] Pytests + CI matrix updated
* [ ] README: "Feature parity" table/flags

---

# Drop-in FSM demo runner (`__main__`)

Append this at the **bottom** of `passive_walker/core/env.py` so you can run the env file directly and watch FSM drive the walker:

```python
if __name__ == "__main__":
    # Quick FSM demo runner for manual inspection / debugging.
    import argparse
    import time
    import numpy as np

    from .io import load_walker_config

    parser = argparse.ArgumentParser(description="PassiveWalkerEnv demo runner")
    parser.add_argument("--config", type=str, default="passive_walker/configs/walker.yaml",
                        help="Path to walker YAML config")
    parser.add_argument("--mode", type=str, default="fsm", choices=["fsm", "research"],
                        help="Environment mode")
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="Sim time limit (s)")
    parser.add_argument("--gui", dest="gui", action="store_true",
                        help="Enable GUI (default)")
    parser.add_argument("--no-gui", dest="gui", action="store_false",
                        help="Disable GUI")
    parser.set_defaults(gui=True)
    parser.add_argument("--seed", type=int, default=None, help="Seed for reset()")
    # Optional: let NN override FSM for quick checks
    parser.add_argument("--nn-hip", action="store_true", help="Use NN action for hip")
    parser.add_argument("--nn-knees", action="store_true", help="Use NN actions for knees")
    parser.add_argument("--print-interval", type=float, default=0.5,
                        help="Telemetry print interval (s)")

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
```

## Run it

* **FSM demo (default):**

```bash
python -m passive_walker.core.env
```

* **Headless (CI/server):**

```bash
python -m passive_walker.core.env --no-gui --seconds 10
```

* **Research mode shaping (FSM control):**

```bash
python -m passive_walker.core.env --mode research
```

* **NN override checks:**

```bash
python -m passive_walker.core.env --nn-hip
```

* **Custom config:**

```bash
python -m passive_walker.core.env --config passive_walker/configs/walker.yaml
```

That's everything merged and ready to implement.
