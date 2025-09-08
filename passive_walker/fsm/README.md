# FSM Data Collection (Ultra-Lean)

This folder provides a **minimal**, **deterministic** finite-state-machine (FSM) data collector using the simplified core environment.

## Quick Start

Collect 2 episodes × 64 steps, headless:
```bash
python -m passive_walker.fsm.collect --episodes 2 --steps 64 --out data/fsm --seed 42
```

Files produced (per episode):

```
data/fsm/
  episode_000000.npz
  episode_000001.npz
  meta.json
```

## Data Schema (NPZ)

Each `episode_*.npz` contains:

* `obs: (T+1, 11)` — observations `[x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]`
* `act: (T, 3)` — actions (zeros in FSM mode)
* `rew: (T,)` — reward per step
* `done: (T,)` — terminal flags
* `info_pitch: (T,)` — |pitch| (radians)
* `info_torso_z: (T,)` — torso height (m)
* `info_dx: (T,)` — forward delta x per step

> Optional (if enabled in the collector):
> `label_qdes: (T, 3)` physical joint targets (FSM desired)
> `label_act:  (T, 3)` the same targets normalized to `[-1, 1]` (BC-ready)

## Determinism

* Set `--seed S`, episodes use `S + episode_idx`.
* Headless, no randomness unless enabled at the top of `env.py`.
* Re-running the same command yields byte-identical arrays.

## Notes

* All parameters live **at the top** of the respective Python files.
* No YAML/config/overrides.
* GUI is off by default in the collector; use the environment module for interactive demos:

  ```bash
  python -m passive_walker.core.env --gui --seconds 5
  ```

## Troubleshooting

* Gym deprecation warnings are harmless; the core uses **Gymnasium**.
* If you run on a headless server and accidentally enable GUI, ensure EGL/OSMesa is available, or run with `--no-gui`.
