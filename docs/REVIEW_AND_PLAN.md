# Passive Walker RL: code review and stabilization plan

> Review written at commit `e0ab85f` and merged with a second, independent review (2026-10-03). The findings in §2.1 items 1–5 were confirmed at runtime; see `docs/baseline.md`. Companion documents: `docs/WALKING_METRIC.md` (benchmark specification) and `docs/ENVIRONMENT.md` (machines and setup).

## Status

| Phase | State |
|---|---|
| 0. Baseline and safety net | **Done.** Baseline numbers in `docs/baseline.md`; golden FSM trajectory test added. |
| 1. Repo hygiene | **Done.** History rewritten (2.35 GB → about 7 MB); dead code removed; `pyproject.toml` packaging; location-independent paths and assets; BC playback import fixed. The pre-rewrite history and the historical experiment artifacts are archived outside the repo (verified git bundle + SHA-256 manifest). |
| 2. Environment correctness | **Next.** |
| 3–9 | Planned (below). |

Work happens on the `stabilization` branch and is merged into `master` at milestones.

## Context

You want to restart `passive_walker_rl` after about a year away (last code commit 2025-10-30; plots added 2026-07-13). Before adding features, the project needs to be understood, stabilized and cleaned up.

This document is the result of a full static review of every active module, the tests, the scripts, the docs, the tracked experiment artifacts and the git history. The critical findings were then reproduced at runtime, and baseline numbers were recorded in `docs/baseline.md` (Phase 0).

**Decisions you made (Q&A):**
- Goal: research / thesis results. Correctness and reproducibility come first. All long-term goals stay relevant, pursued in sequence:
  1. reproduce the FSM;
  2. outperform it with learned policies;
  3. compare learning algorithms;
  4. eventually transfer to physical hardware, which is blocked until a hardware specification exists.
- Keep **both** BC backends (PyTorch and JAX/Equinox), at feature parity.
- Full cleanup with a history rewrite. **Done**, with the old history archived outside the repo.
- Breaking the env/obs interface is fine. Old checkpoints and datasets become obsolete; we retrain.
- In scope: BC fidelity vs the FSM, BC→PPO fine-tuning, recurrent vs feedforward policies, robustness (domain randomization and pushes).
- Rewrite PPO in-house, CleanRL-style.
- Reward speed target: you don't remember. Make it configurable and measure the FSM's reward under each option before choosing.
- Evaluation combines survival, distance, speed tracking, actuation efficiency, gait quality and robustness into one versioned benchmark (`docs/WALKING_METRIC.md`), with every component also reported on its own.
- Tracking: TensorBoard plus a JSON/CSV run directory.
- Machines: development on an Ubuntu 24.04 PC; training on the UT Dallas Juno servers (details pending). See `docs/ENVIRONMENT.md`.
- Repository rule: no tool or assistant attribution in commits, trailers, branch names or file contents.

**Design decisions to preserve.** These are intentional and must not be "fixed":
- the knees are sliders (units are m and m/s);
- the leg structure is asymmetric (left leg fixed to the torso, a single relative hip hinge on the right leg);
- the slope is tilted gravity on a flat plane;
- policy actions are PD *targets*, not torques;
- the FSM targets are discontinuous (smoothing them broke balance).

---

## 1. Understanding the project

### Purpose
The project learns walking controllers for a **Variable-Length-Leg (VLL) passive bipedal walker** in MuJoCo:
1. A hand-designed **FSM** controller walks stably. It is the expert.
2. **FSM demonstrations** are recorded to NPZ files.
3. **Behavior Cloning (BC)** imitates the FSM. It can control hip only, knees only, or both; the rest stays under FSM control ("hybrid" modes).
4. **PPO** trains policies, from scratch or (intended) initialized from BC, to match or beat the FSM.

### Physical model (`passive_walker/assets/passiveWalker_model.xml`)
- A planar torso with 3 DOF (`slide_x`, `slide_z`, `pitch`).
- The **left leg is rigidly attached to the torso**. The **right leg swings on a single `hip` hinge**. The hinge has no range limit and damping 0.1.
- Each leg has a **prismatic "knee"** (slider, ±0.5 m, damping 1) that shortens the leg for foot clearance.
- A flat plane. The ramp is simulated by **tilting gravity** by `RAMP_DEG = 10°` (`env.py:209`).
- 1 kHz physics (`timestep=0.001`) and 100 Hz control. **PD control runs in Python** on every 1 ms micro-step (`env.py:416-427`). The actuators are pass-through `general` actuators with ctrlrange ±50 N·m (hip) and ±800 N (knees).

### Control and data flow
```
FSMStateMachine (core/controller.py) ─┐
                                      ├─> PassiveWalkerEnv.step (core/env.py): qdes → Python PD @1kHz → MuJoCo
NN action in [-1,1] → PDController.denorm ┘        modes: fsm | research | hybrid_hip | hybrid_knees
        │
fsm/collect.py ──> episode_XXXXXX.npz (obs, act, rew, done, info_qdes, info_fsm_*, …)
        │
bc/training/train.py (torch MLP | jax MLP | temporal LSTM/GRU, no CLI) ──> .pt/.eqx + meta.json
        │
bc/evaluation/play.py, tools/evaluation/bc_comprehensive_eval.py ──> closed-loop eval
        │
ppo/train.py + trainer.py + models.py (MLP/LSTM/GRU actor-critic) ──> final_model.pth
```

- **FSM logic.** The hip toggles between ±0.5 rad when the stance foot is in contact and the swing leg is behind. Each knee retracts to −0.25 m at heel-strike and releases once the leg has swung 0.10 rad forward. Contact means foot z < 6 cm. All targets are **bang-bang** (discrete), as your `docs/fsm_analysis` plots show.
- **Observation (17-D).** `[x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇, l_contact, r_contact, l_force, r_force, l_dur, r_dur]`. The first 11 dims were the original v1 observation; the 6 contact dims were added later.
- **Action (3-D)** in [−1, 1] is mapped linearly to joint targets of ±0.5 (rad for the hip, m for the knees).
- **Rewards (`core/reward.py`).** A simple "fsm" reward (also used in the hybrid modes) and a 7-term "research" reward. Termination is |pitch| > 1.3 rad or torso z < 0.70 m.

### Module status map
| Area | Files | Status |
|---|---|---|
| Env / FSM / PD | `core/env.py`, `core/controller.py`, `core/reward.py`, `core/randomization.py` | Works, with **correctness bugs** (section 2.1) |
| FSM collection | `fsm/collect.py` | Runs, but the data is degenerate (identical episodes, zero `act`) |
| Curriculum collection, perturbations, physics conditions | `fsm/curriculum_collect.py`, `core/perturbations.py`, `core/physics_conditions.py` | **Non-functional** (no-ops or crashes) |
| BC training | `bc/training/train.py`, `bc/data/*`, `bc/models/*` | MLP training works with `--label-type qdes`. The default (`act`) and temporal training are broken. |
| BC play / eval | `bc/evaluation/play.py`, `evaluate.py` | Playback works again (import fixed in Phase 1). The comprehensive evaluator is broken (Phase 6). |
| PPO | `ppo/train.py`, `trainer.py`, `models.py`, `buffer.py` | **The CLI crashes.** The trainer has algorithmic bugs. The past models came from the archived `enhanced_trainer.py`. |
| Unused | `passive_walker/jax/*`, `core/environment_enhancements.py`, `deployment/` (2 identical 474-line files), `bc/training/schedulers.py`, `bc/data/curriculum.py`, `ppo/ppo_train.yaml`, `config/paths_redirect.py`, `_archive/` (160 files) | **Removed in Phase 1.** `core/controller_jax.py` (the opt-in slow PD path) goes in Phase 2. |
| Tools / scripts | `tools/*.py`; `scripts/overnight_*.sh` | Tools are partly broken (missing functions, wrong defaults). The overnight scripts deleted their own outputs and were **removed**; Phase 6 replaces both. |

---

## 2. Problems found

### 2.1 Critical correctness bugs (they invalidate results)

1. **BC labels are all zeros by default.**
   - The collector stores `act_buf[t] = action`, and that action is always zeros (`fsm/collect.py:324,352`). The real FSM targets live only in `info_qdes`.
   - `--label-type` defaults to `act` (`bc/training/train.py:1013`).
   - Temporal training hard-codes `label_type="act"` (`train.py:565,579,790,804`).
   - `scripts/overnight_bc_sweep.sh` uses the default.
   - So every `act`-labelled model learned to output the constant 0. The tracked `torch_*_seed123_ep1_steps9000` metrics confirm it: val loss goes to about 0.001.
   - Only the `qdes` models (`experiments/models/bc/*steps180000*`) learned anything real.
2. **All FSM episodes are identical.**
   - `reset()` is fully deterministic: there is no initial-state noise, and physics is randomized only when `randomize_physics=True` (`env.py:242-277`).
   - In `experiments/data/fsm_runs/README.json`, all 20 episodes have exactly the same distance (6.7200532 m), and your overview plot shows episodes 0–2 overlapping exactly.
   - So a dataset of N episodes is one trajectory repeated N times, and the episode-wise train/val split leaks.
3. **The contact observation is wrong.**
   - `mj_contactForce` returns its result in the contact frame, where index 0 is the **normal** force. The code reads `force[2]`, a tangential (friction) component (`env.py:377`).
   - So `l/r_force` and the force-thresholded contact flags are physically wrong.
   - The contact durations also add the 1 ms physics timestep once per 10 ms control step (`env.py:316`), so they run 10× too slow.
4. **PPO is broken end to end.**
   - `ppo/train.py` calls `torch.manual_seed` without importing torch (line 150) and calls `trainer.train()` with no env (line 179). Either one crashes it.
   - Value clipping uses `old_log_probs` where the old values belong (`trainer.py:181-185`).
   - Each rollout resets the env (`trainer.py:323`), and GAE bootstraps with 0 (`trainer.py:228`). With n_steps 2048 and 2500-step episodes, **episodes never finish**.
   - A time limit is treated as a terminal state.
   - The Gaussian actions are unbounded and unclipped, and the hip has no joint range, so targets can be arbitrary.
   - There is no observation normalization, even though absolute x grows to about 36 m.
   - LSTM/GRU **never receive a hidden state during training** (`trainer.py:333`), although playback does carry one. "Recurrent PPO" has been an MLP.
   - `load_bc_weights` reads the keys `"weight"`/`"bias"` (`ppo/models.py:516`), which don't exist in a BC state_dict. **BC→PPO initialization cannot work.**
5. **BC evaluation is broken.**
   - `bc/evaluation/__init__.py:4` imports `evaluate_model`, which doesn't exist. So `python -m passive_walker.bc.evaluation.play` fails at import.
   - `ComprehensiveEvaluator` imports a `create_model` that doesn't exist (`evaluate.py:204,230`) and runs the env in **`fsm` mode by default**, where actions are ignored (`evaluate.py:250`). It evaluates the FSM, not the model.
   - `tools/evaluate_model.py` and `tools/compare_models.py` import `evaluate_bc_model` and `evaluate_ppo`, which don't exist.
   - Different tools run hip-only models in different env modes: `play.py` uses the hybrid modes, while `bc_comprehensive_eval.py` always uses `research`.
6. **The robustness features are no-ops.**
   - `PerturbationManager` schedules times but never creates a perturbation.
   - `env.model.body_names` doesn't exist (`perturbations.py:347`).
   - The terrain and mass "changes" set attributes the env never reads, and `xfrc_applied` is never cleared.
   - `curriculum_collect.py` passes an invalid mode and the same seed to every stage, and `json.dump`s Enum keys, which crashes.
   - `physics_conditions.py:341` uses `mujoco` without importing it, and its `*=` scaling compounds across episodes.
7. **CLI flags are silently ignored.**
   - The env CLI's `--ramp-jitter`, `--friction-*` and `--randomization-profile` do nothing, because `randomize_physics` is never set.
   - PPO's `--use_domain_randomization`, `--use_curriculum` and `--randomization_profile` are unused, and the profile `light` isn't valid anyway.
   - `--perturbation-freq` is unused.
   - Hybrid-mode collection sends a zero NN action, which produces degenerate data.
8. **The reward design doesn't fit the expert.**
   - The research reward targets 0.15 m/s. The FSM walks at about **1.4 m/s** (normalizer mean ẋ = 1.44).
   - The symmetry term penalizes the alternating knee retraction the gait needs.
   - The upright bonus is about 0 at the FSM's pitch (std about 0.33 rad).
   - The foot-clearance softplus is nearly constant.
   - The research `w_pitch` weight is configured but never used.
   - The effort term adds hip torque (N·m) and knee forces (N) as if they had the same unit.
   - Per-step terms are not scaled by the control period.
   - A formula-only check gives idle, upright standing about 0.84 reward per step.
   - `jax/reward_jax.py` used a different, incompatible weight schema (now removed).
9. **Observation aliasing silently corrupts PPO transitions (critical).**
   - `env._get_obs()` fills and returns the same reused `self._obs` buffer on every call (`core/env.py:283,307`).
   - PPO calls `env.step()` first and then `buffer.add(obs, …)` (`ppo/trainer.py:349,352`). By then `obs` already holds the *next* state, so every stored state is paired with an action chosen from a different state.
   - BC playback appends the same buffer to its frame-stack history, and the `rollout_obs` test helper keeps an uncopied reset observation.
   - The collector is unaffected, because it copies into preallocated arrays.
   - Fix: the env returns a fresh array (Phase 2), plus a transition-alignment test (Phase 5).
10. **PPO evaluation and logging never fire.**
    - `train()` checks `timestep % eval_freq == 0` and `timestep % log_freq == 0` while `timestep` grows in steps of 2048 (`ppo/trainer.py:463,485`).
    - With the CLI's default `eval_freq = timesteps // 20` (5000 for 100k), the first trigger is at 1.28M steps. For 25k it's 6.4M, and the 1000-step log interval first fires at 256k.
    - None of the 16 historical TensorBoard files contain evaluation scalars.
    - Fix: threshold-based scheduling (Phase 5).
11. **Randomization is not reproducible.**
    - The `DomainRandomizer` is built once and keeps its original RNG after the env is reseeded.
    - A profile alone doesn't enable randomization (item 7).
    - `physics_conditions.py` implements "stiffness" by changing armature.
    - Fix: explicit RNG ownership, reset from an immutable baseline model, and per-episode logging of the physics actually realized (Phase 2).
12. **Contact bookkeeping has side effects.**
    - Contacts are aggregated over the foot body's geoms without filtering the other geom to the ground.
    - Reading the observation advances the contact-duration state, so calling `_get_obs()` twice changes the result.
    - Fix in Phase 2.
13. **Temporal BC losses and augmentation are inconsistent.**
    - The masked loss averages over padded positions, so its magnitude depends on padding.
    - The Torch and JAX helpers normalize differently, and one JAX helper masks after reducing to a scalar.
    - Temporal training calls augmentation with three Torch tensors, but the augmentation interface takes two NumPy arrays, so enabling augmentation crashes.
    - The BiLSTM uses future observations, so it can't be used as an online controller.
    - Fix in Phase 4.
14. **Evaluation and tools have further breakages.**
    - The BC evaluation CLI passes a `seed` that `EvaluationConfig` doesn't define.
    - The PPO evaluator builds its env without a mode, so it defaults to `fsm` and ignores the policy.
    - Standard BC evaluation skips the saved normalizer.
    - FSM comparison statistics are hard-coded, "imitation error" is measured against zero actions, and "energy" uses normalized target magnitudes.
    - Success is defined differently across tools (50 steps, 100 steps, 80% of the horizon).
    - `tools/visualize_results.py` imports a nonexistent Torch `SummaryReader`, and `tools/compare_models.py` defines `--output` but reads `args.out`.
    - The env's GUI fallback can't catch window failures, because the window is created lazily on the first `render()`.
    - Fix: replaced by the Phase 6 harness.
15. **The overnight BC sweep deletes data.** `scripts/overnight_bc_sweep.sh:675` runs `find "$RUN_DIR" -type f ! -name "*.png" … -delete`, which removes every checkpoint (`.pt`, `.eqx`) and dataset (`.npz`) it just produced. The script and its PPO sibling have been removed.
16. **The historical FSM plots disagree with each other.** `docs/fsm_analysis/view_plots.html` reports 35.9 m per episode, while `fsm_quality_metrics.png` and the overview's cumulative-distance panel show about 441 m. The plotting code summed positions instead of displacements. Treat those plots as qualitative only.

### 2.2 Architecture, maintainability and repo hygiene
- **Not a real Gymnasium env.**
  - `step` returns a 4-tuple, the metadata uses the old key, and there's no `render_mode`/`rgb_array` and no registration. So `gymnasium.vector` and the env checker can't be used.
  - `info["seed"]` is attached on every step (`env.py:491`).
- **Packaging.**
  - `XML_PATH` is relative to the working directory (`env.py:16`), so everything breaks outside the repo root, and the XML isn't in `package_data`.
  - `common/` and `bc/models/` have no `__init__.py`.
  - `setup.py` parses its own *generated*, tracked `egg-info`. The `walker-train-bc` entry point points to a module that doesn't exist.
  - Versions disagree: 0.1.0 vs 2.1.0.
  - `pytest` is a runtime dependency, while equinox, optax and tensorboard are missing.
- **Duplication.**
  - The joint ranges and PD gains are copied in 5+ places.
  - `_assemble_action` is copy-pasted 4×, and `play_torch`/`play_jax` are near-duplicates.
  - There are two JAX controller modules.
  - There are two config systems. The BC dataclasses aren't used by the argparse CLI, and the PPO YAML is stale.
  - Torch and JAX checkpoint metadata use different keys (`input_dim` vs `in_dim`, different normalizer layouts).
- **Learning design issues.**
  - The observation contains absolute `x`, which goes out of distribution past the training distance.
  - BC regresses discrete bang-bang targets with a tanh head. Hip labels sit exactly at ±1, so tanh saturates.
  - The `both-adv` smoothness loss is computed on *shuffled* batches (Torch path).
  - `play.py` hard-codes hidden=512, keeps the frame-stack buffer as a function attribute that is never reset, and ignores the meta's `frame_stack`.
  - In the overnight sweep, mlp_small and mlp_large write the same filename, and `|| true` hides failures.
- **Side effects.** `config/paths.py` creates about 15 directories on import.
- **Tests.**
  - There are 5 files, and most wrap their body in `try/except → pytest.skip`, which hides failures.
  - Nothing covers the collector's content, BC training, play/eval or the PPO loop, and there's no CI.
- **Docs are fictional or stale.**
  - The README is a stub.
  - `docs/API.md` documents APIs that don't exist.
  - The docs link to TRAINING.md and CHANGELOG.md, which don't exist.
  - The core README says the observation is 11-D.
  - `scripts/README.md` describes scripts that don't exist.
  - The Makefile `demo` uses flags that don't exist.
  - `.gitignore` ignores `experiments/` and `.*`, but files under both are tracked.
- **Repo size.**
  - `.git` is 2.35 GB, almost all of it ~3 GB of `results/brax/sweep_results/*.msgpack` blobs (63 MB each) from history.
  - About 29 MB of artifacts are tracked in the tree (`.pth` files, tfevents, PNGs, egg-info).

### 2.3 Performance bottlenecks (estimated; measured in Phase 0)
- **Python micro-step loop.**
  - Each control step runs 10 micro-steps, and each one does a numpy PD computation, 3 ctrl writes, `mj_step`, and list-to-array copies.
  - Fix: move PD into MuJoCo (`gainprm=kp`, `biasprm=[0,-kp,-kd]`, `forcerange`) and call `mj_step(m, d, nstep=10)`. This should give several times the throughput. It is mathematically identical, because PD uses q/q̇ at each micro-step either way; we'll verify equivalence numerically.
- **Contacts.** `_get_contact_force` rescans every geom on each call and allocates per contact. Fix: `<touch>`/force sensors on foot sites, or precomputed geom sets.
- **JAX PD path.** It is 15–20× slower per step (by its own docstring). Remove it from the env; JAX stays in BC and in the future MJX work.
- **PPO.** It uses a single env on a 4-core machine. Use `AsyncVectorEnv`, which needs the Gymnasium API fix first.
- **BC.**
  - Tensors are rebuilt every epoch.
  - Frame stacking and observation noise use per-element Python loops. Fix: `sliding_window_view` and vectorized noise.
- **GUI.** `swap_interval(1)` ties sim speed to vsync, and there's no real-time sync.

---

## 3. Prioritized stabilization plan

Each phase ends green: tests pass, and it gets its own commit(s) on the `stabilization` branch.

### Phase 0: Baseline and safety net (no behavior change)
- Create a reproducible environment: `pyproject.toml` extras `[torch]`, `[jax]`, `[dev]`, plus a pinned lockfile or `requirements-dev.txt`. Install it in the container.
- Record baselines:
  - the current test results;
  - env control-steps/s with the NumPy PD;
  - FSM 25 s rollout metrics (distance, speed, falls, gait cycles, cost of transport);
  - BC epoch time on a small dataset.
  - Save them to `docs/baseline.md` and use them as the before/after reference.
- Add a **golden FSM trajectory test**: a 5 s FSM rollout checked against stored hip/knee targets and x(t), within tolerance. This guards every env refactor.

### Phase 1: Repo hygiene (gated history rewrite)
- **History rewrite:**
  1. Make a `git clone --mirror` backup.
  2. Run `git filter-repo` to strip `results/`, `*.msgpack`, `*.pkl`, `*.pth`, `*.eqx`, tfevents and egg-info from all history.
  3. Expected result: about 10–30 MB.
  4. **Force-pushing `master` and the branches needs your explicit confirmation at that moment.** Anyone with a clone must re-clone.
- Delete `_archive/` (it stays recoverable from the backup and tags), `deployment/`, `passive_walker/jax/`, `core/environment_enhancements.py`, `bc/training/schedulers.py`, `bc/data/curriculum.py`, `ppo/ppo_train.yaml`, `config/paths_redirect.py`, the tracked egg-info and the root PNG.
- Untrack `experiments/` and fix `.gitignore` (stop ignoring `.*` wholesale). Keep `experiments/README.md`.
- Packaging:
  - Replace `setup.py` with `pyproject.toml` and a single version source.
  - Add `package_data` for the XML and the missing `__init__.py` files.
  - Use correct console scripts: `walker-collect`, `walker-train-bc`, `walker-train-ppo`, `walker-play`, `walker-eval`.
  - Resolve the XML path via `importlib.resources`.
- Make `config/paths.py` side-effect-free: create directories lazily, with the root overridable through `PASSIVE_WALKER_HOME`.

### Phase 2: Environment correctness (obs schema v2)
- Move to the Gymnasium API:
  - `step` returns `(obs, r, terminated, truncated, info)`;
  - `TimeLimit`-style truncation;
  - `render_mode` with "human" and "rgb_array" (via `mujoco.Renderer`);
  - register `PassiveWalker-v2`;
  - pass `gymnasium.utils.env_checker`.
- Use Gymnasium seeding (`super().reset(seed=…)`, `self.np_random`) as the single RNG owner. Every random component (physics randomization, initial state, pushes) draws from it, so a reseed resets all of them (§2.1 item 11).
- **Return a fresh observation array** from `reset()` and `step()`; never the internal buffer (§2.1 item 9).
- Centralize all constants in one `core/params.py` (or a dataclass): joint ranges, gains, limits and FSM thresholds. Everything else imports from there.
- Fix contacts:
  - read the normal force (`force[0]`) or use touch sensors;
  - count only foot–ground contacts;
  - accumulate durations with the control dt, and update them in `step()`, not in the observation getter;
  - share one contact definition with the FSM, or document why they differ (height vs force).
- **Observation v2.**
  - Drop absolute `x`. Keep z, pitch, the velocities, the joint states and the fixed contact features. Make the layout explicit in a schema with named indices.
  - Add `obs_version` to `info` and to every artifact.
- Add initial-state randomization behind a config: a small jitter on hip angle, torso pitch and velocities, plus an optional random FSM phase. Seeded resets then actually differ.
- Make randomization explicit:
  - one `PhysicsConfig` (ramp, friction, mass, damping, gain jitter) that applies whenever a profile is given;
  - restore nominal values on reset;
  - fix the ignored CLI flags.
- Clip actions to [−1, 1] in the env, and add a hip range in the XML.
- Make the reward configurable.
  - Turn the research reward's speed target, symmetry term and upright band into config parameters.
  - Add a `scripts/score_fsm_reward.py` that reports the FSM's per-term reward under each preset, so we can **decide the target from data**. This is the open question you flagged.
  - Fix the ineffective terms (§2.1 item 8): use or drop `w_pitch`, normalize hip and knee effort separately, and scale per-step terms by the control period.
  - Keep the training reward separate from the benchmark metric (`docs/WALKING_METRIC.md`). The metric doesn't change when the reward is tuned.
- The same report records the nominal FSM reference row for the walking metric: net distance, speed, actuation work from MuJoCo `actuator_force × actuator_velocity`, pitch RMS, and contact timing.
- Remove the JAX PD path from the env.

### Phase 3: Data collection correctness
- Collect the FSM's normalized `qdes` as **the** label. Write it under a single, explicit `act` key, and drop the zeros. Add `meta.json` fields: `obs_version`, `label_space`, the physics actually realized for each episode (ramp, friction, mass) and the seeds.
- Collect diversity through initial-state jitter plus physics profiles. Add a dataset QA check that **fails** on duplicate episodes.
- Rewrite `core/perturbations.py` as a small, working push module:
  - timed impulses or pushes on the torso via `xfrc_applied`, cleared after their duration;
  - Poisson or scheduled timing;
  - a seeded RNG;
  - pushes logged per step.
  - It is shared by collection, evaluation and PPO.
- Delete `curriculum_collect.py` and `physics_conditions.py`. Replace them with `collect --profile …` over the fixed randomization and push config.
- Hybrid-mode collection is unnecessary, because hybrid BC trains on FSM data with section masks. Restrict the collector to `fsm`.
- Vectorize the observation-noise augmentation.

### Phase 4: BC pipeline (Torch and JAX at parity)
- One config dataclass (YAML-loadable) consumed by the CLI.
  - Remove the default `--label-type act` trap: labels come from the dataset's declared label space.
  - Add a CLI entry for the temporal models.
- Shared, backend-agnostic pieces:
  - dataset loading (with `sliding_window_view` frame stacking);
  - the normalizer;
  - the **checkpoint and metadata schema**: one `meta.json` with backend, arch, dims, normalizer, obs_version, section, frame_stack and data hash;
  - action assembly.
  - Only the model and training-step code differ per backend.
- Fixes:
  - temporal training gets input normalization and real labels;
  - one masked temporal loss, averaged over valid steps only and shared by Torch and JAX;
  - the augmentation interface matches the temporal batches;
  - BiLSTM is kept for offline analysis only, never as a controller;
  - no shuffling when a smoothness loss is used (or compute it inside sequences);
  - `play` builds the model from meta (no hard-coded sizes) and resets the frame-stack buffer per episode;
  - JAX dropout is actually applied.
- Address tanh saturation: label smoothing such as ±0.95, or a linear head plus clipping. Also offer a **classification head** (FSM states → targets) as a variant, since the targets are discrete.
- Add a **parity test**: the same tiny dataset trained on Torch and on JAX should reach comparable loss and identical meta schema.

### Phase 5: PPO rewrite (in-house, CleanRL-style)
- Vectorized envs (`gymnasium.vector.AsyncVectorEnv`, N = cores).
- Running obs and return normalization, saved inside the checkpoint.
- Correct GAE with value bootstrap at the rollout end and terminated vs truncated handling, plus correct value clipping (old values).
- Bounded actions with consistent probability math. Either use a tanh-squashed Gaussian with the log-det-Jacobian correction, or sample unclipped and clip only when executing (CleanRL style), with log-probs computed on the sampled action. Document the choice.
- A state-independent log-std, and separate actor and critic networks (optionally shared).
- Store the observation that was used to choose the action, snapshotted before `env.step` (§2.1 item 9). A unit test checks `(obs_t, a_t, r_t, obs_t+1)` alignment with distinguishable states.
- Threshold-based evaluation and logging schedules ("next eval at ≥ N steps"), not exact modulo (§2.1 item 10).
- Check correctness against a trusted reference (CleanRL PPO on Pendulum) before any walker results.
- **Recurrent PPO done properly:** hidden state carried through rollouts and reset on done, plus sequence-chunked minibatches with stored initial hidden states (LSTM and GRU).
- **BC→PPO initialization:**
  - the actor architecture matches the BC MLP/temporal model, and weights load through the shared checkpoint schema;
  - the critic warms up while the policy is frozen for K updates;
  - an optional BC regularizer (KL or L2 to the BC policy, annealed).
- Resume from checkpoint, deterministic eval on a separate env set, and a best-model checkpoint by eval score.
- Logging: TensorBoard plus `runs/<exp>/<timestamp>/{config.yaml, metrics.csv, eval.json, ckpt/}`.
- Delete `ppo/buffer.py`'s unused vector buffer, `evaluate_cli.py`/`evaluate.py` duplication and `plot_ppo_results.py`. They fold into Phase 6.

### Phase 6: One evaluation harness and the walking benchmark
- `passive_walker/eval/`, a single `evaluate(policy, env_cfg, conditions, seeds)` used for FSM, BC and PPO alike.
- **Policy adapters** (FSM, hybrid-hip BC, hybrid-knees BC, full BC, PPO) own preprocessing, control mode, action assembly, model reconstruction from metadata, and recurrent-state reset.
- An **action-sensitivity test** proves that two materially different policies produce different trajectories, so the harness can never silently evaluate the FSM.
- It implements **Walking Metric v0** from `docs/WALKING_METRIC.md`:
  - per-episode raw measurements: full-horizon survival, net distance, speed-tracking RMSE, actuation cost of transport from actuator force × velocity, and gait quality;
  - per-condition scores `Q_c`;
  - the **Nominal Walker Score** and **Robust Walker Score**;
  - the qualification badge and a result card.
- Imitation error is measured against the FSM's target *in the same state*.
- Output is a per-episode JSON/CSV plus standard plots, stamped with the metric version, model-XML hash, git SHA and seeds.
- The FSM is always evaluated as the reference row.
- Report N ≥ 5 seeds with confidence intervals (rliable-style IQM / bootstrap at the run level).
- Simulated time and wall-clock time are always recorded separately.
- Replace `tools/*` and the deleted overnight shell scripts with `walker-sweep` (a YAML grid run through Python), with no hard-coded paths, no `|| true` and no deleting cleanup steps. Every run writes to a unique directory.

### Phase 7: Performance
- MuJoCo-native PD and `nstep`, checked against the golden trajectory test.
- Touch sensors for contacts.
- Profile with `cProfile`, and benchmark with `scripts/bench_env.py` (steps/s for 1 env and the vector env).
- Target: about 2.5–3× single-env throughput over the Phase 0 baseline. `mj_step` is about 26% of step time, which caps a single env at roughly 17k control steps/s. Further gains come from vectorized envs (roughly ×cores) and, longer term, MJX. Report the actual numbers.

### Phase 8: Tests, CI and docs
- Tests:
  - no `try/except skip`;
  - units: params, obs schema, PD, the FSM transition table, reward terms, normalizer and checkpoint round-trips;
  - integration: collect 2 eps → BC 1 epoch (Torch and JAX) → play 1 s → PPO 2 updates → eval.
  - Mark slow tests; they run from any working directory.
- GitHub Actions: lint (ruff), the fast test suite, and a slow suite on a nightly or manual trigger.
- Rewrite the README (purpose, install, a 5-command quickstart, results table), `docs/ARCHITECTURE.md` (this design), `docs/DATA.md` (NPZ schema v2) and `docs/EXPERIMENTS.md`. Delete the fictional API.md.
- Add a CHANGELOG starting at v3.0.0, since this is a breaking release.

### Phase 9: Cluster readiness (UT Dallas Juno)
Juno is a Slurm cluster; see `docs/ENVIRONMENT.md`. Current training is CPU work, so it runs on the `normal`/`dev` CPU partitions (64 cores, 384 GB per node). GPU partitions are used only once the MJX/JAX phase starts. Planned:
- Slurm templates for `dev` (smoke) and `normal` (training and sweeps), with explicit `-c`, `--mem` and `-t`, and an environment check at job start. Every submission needs the owner's approval.
- Headless operation everywhere: no GUI imports on the training path, `MUJOCO_GL` unset (or `egl` only when rendering videos).
- `PASSIVE_WALKER_HOME` pointed at cluster scratch, so experiment outputs never land in the code checkout.
- One job = one run directory, with:
  - config;
  - seed;
  - git SHA and dirty flag;
  - `pip freeze`;
  - model-XML hash;
  - host info.
- Job templates for single runs and multi-seed array jobs, plus a small aggregation command that turns a sweep directory into the result card.
- Thread pinning (`OMP_NUM_THREADS`, `torch.set_num_threads`) matched to allocated cores; vectorized PPO sized to the allocation.
- Resumable checkpoints so preempted or time-limited jobs can continue.

---

## 4. Longer-term roadmap (after stabilization)

**A. Thesis experiment suite** (each one is a sweep config plus a results notebook)
1. BC fidelity: section (hip / knees / both) × architecture (MLP / LSTM / GRU) × dataset size and diversity → closed-loop success and imitation error.
2. **DAgger.** The FSM can be queried in any state, so DAgger is the natural fix for BC compounding errors. It is likely the strongest BC result.
3. PPO from scratch vs BC-initialized vs BC-regularized: sample efficiency, final return, CoT, robustness grid.
4. Recurrent vs feedforward under partial observability (drop velocities or contacts, add obs noise or latency).
5. Robustness: train with domain randomization and pushes, test on unseen slopes, friction and pushes. Measure push-recovery success vs impulse.

**B. Method extensions**
- Residual RL on top of the FSM (policy = FSM + learned correction). It is a strong, interpretable baseline.
- An asymmetric actor-critic: the critic gets privileged physics parameters.
- Speed-conditioned policies (target velocity as input), which also resolves the reward-speed question cleanly.
- An energy-efficiency objective (cost of transport) and Pareto curves of speed vs CoT.

**C. Simulation fidelity**
- Real sloped or heightfield terrain (steps, varying slope) instead of tilted gravity.
- Actuator models: torque/velocity limits, motor dynamics, control latency, sensor noise.
- Optionally a 3D or full-model variant.

**D. Scale**
- A MuJoCo **MJX + JAX** backend for thousands of parallel envs (it uses the JAX side you're keeping), with the BC/PPO JAX paths sharing models.
- Hydra configs and multirun.

**E. Hardware transfer (eventual goal; blocked on a hardware specification)**
- Define the actual robot: geometry, sensors, actuators and their units, achievable control rate.
- System identification of the physical VLL walker, then calibration of the simulated model and actuator/sensor transfer functions.
- Model latency, saturation, sensor noise and electrical energy. Mechanical work in simulation is not battery consumption.
- Real-time inference (p95 and worst-case latency), a watchdog, a safety supervisor with the FSM as fallback, and staged hardware validation.
- Policy distillation to a small MLP for deployment.

---

## Critical files to change
`passive_walker/core/{env.py,controller.py,reward.py,randomization.py,perturbations.py}`, `passive_walker/assets/passiveWalker_model.xml`, `passive_walker/fsm/collect.py`, `passive_walker/bc/{training/train.py,data/dataset.py,models/*,evaluation/*,utils.py,config.py}`, `passive_walker/ppo/*` (rewrite), new `passive_walker/eval/`, `passive_walker/config/paths.py`, `setup.py` → `pyproject.toml`, `tests/*`, `README.md`, `docs/*`, `scripts/*`, `tools/*`.

Reused as-is or with light edits: the `FSMStateMachine` transition logic (`core/controller.py:223-319`), `Normalizer` (`bc/utils.py`), the Torch and JAX MLP/temporal model classes (`bc/models/*`), the actor-critic modules as a starting point (`ppo/models.py`), and `PHYSICS_PRESETS` (`fsm/collect.py:48`).

## Verification (per phase and end to end)
- `pytest -q` (fast) and `pytest -m slow` all green, with no skips that hide errors. `gymnasium.utils.env_checker.check_env` passes.
- The golden FSM trajectory test passes after every env or performance change. FSM 25 s success is 100% on nominal physics, and the speed is reported.
- Dataset QA: no duplicate episodes; labels are non-constant and lie in [−1, 1].
- The smoke pipeline runs from a clean clone outside the repo root: `walker-collect` → `walker-train-bc --backend torch|jax` → `walker-play --no-gui` → `walker-train-ppo` (tiny) → `walker-eval`.
- PPO sanity: it learns Pendulum-v1 or a short walker task within budget, and its values are bootstrapped (unit test on GAE with truncation).
- Benchmarks: `scripts/bench_env.py` before and after, reported in `docs/baseline.md`.
- History rewrite: done and verified (fresh clone about 7 MB). The pre-rewrite history is archived as a verified git bundle outside the repo.

## Open decisions
- The speed target(s) and the final reward weights are chosen after the Phase 2 FSM reward-scoring report.
- The walking-metric parameters (task speed, tracking tolerance, scenario matrix, condition and component weights, qualification thresholds) are frozen as metric v1 after the FSM reference measurement. See `docs/WALKING_METRIC.md`.
- Juno layout for this project (code and env under `~/work`, runs on `~/scratch`) is proposed in `docs/ENVIRONMENT.md` and still needs the owner's confirmation.
- The target hardware platform, if and when hardware work starts.
