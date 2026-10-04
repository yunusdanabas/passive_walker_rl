# Baseline (before stabilization)

Recorded on 2026-10-04 at commit `e0ab85f`, before any Phase 1+ changes. This is the
reference point for the stabilization work in `docs/REVIEW_AND_PLAN.md`.

**Machine:** 4 vCPU Intel Xeon @ 2.10 GHz, no GPU, Python 3.11.
**Packages:** see `requirements-dev.txt` (mujoco 3.14.0, torch 2.14.1 CPU, jax 0.10.2).

## Tests

`pytest -q`: 14 passed. Two notes:
- The original `tests/conftest.py` forced `MUJOCO_GL=egl`. On machines without EGL,
  `import mujoco` then fails at collection time. That line is removed.
- The suite does not exercise the broken paths listed below. Most tests also wrap their
  body in `try/except → pytest.skip`, which would hide failures.

A golden FSM trajectory test was added (`tests/test_fsm_golden.py`). Tiny numerical
perturbations (float32 JAX PD, a 1e-9 m/s velocity offset) change 5 s trajectories by
less than 2e-7, so the 1e-4 tolerance only trips on real behavior changes.

## Environment throughput (`python scripts/bench_env.py --profile`)

| Metric | Value |
|---|---|
| Control steps/s (FSM mode, NumPy PD, 1 env) | 4.6k–6.6k (run-to-run variance) |
| Physics steps/s | 46k–66k |
| Share of time in `mj_step` | about 26% |

The rest of the time is Python overhead: the per-micro-step PD, ctrl writes, list-to-array
copies, quaternion→Euler conversion in the FSM, and the contact scan. That overhead is the
target of Phase 7. `mj_step` alone caps a single env at roughly 17k control steps/s with
this XML, so a realistic single-env gain is about 2.5–3×. Larger gains need vectorized envs.

## FSM reference behavior (nominal: ramp 10°, friction 0.9)

| Metric | Value |
|---|---|
| 25 s episode, seeds 0–4 | 2500 steps, **35.948579 m**, mean ẋ = **1.438 m/s**, no falls |
| Distinct trajectories across seeds | **1** (reset is fully deterministic) |
| Gait cycles per 25 s | about 16 |

## Data and training timings

| Task | Time |
|---|---|
| Collect 10 × 25 s FSM episodes | 4.9 s wall |
| BC Torch MLP-large, 22.5k samples, 3 epochs (`--label-type qdes`) | 5.3 s wall (val L1 0.019) |
| BC JAX MLP, same data, 3 epochs | 4.6 s wall (val L1 0.036) |

## Runtime-confirmed defects (details in REVIEW_AND_PLAN.md §2.1)

| Check | Result |
|---|---|
| Collected `act` labels | all zeros. `info_qdes` takes the values {−0.5, −0.25, 0, 0.5} |
| Episodes with different seeds | identical observations |
| Contact force used by env (`force[2]`) | mean 12.9 N. The normal force (`force[0]`) is 71.25 N, about the total weight of 71.6 N |
| Contact-duration obs after 3 s of walking | 0.027 s (expected: seconds) |
| `import passive_walker.bc.evaluation` (and therefore `…play`) | `ImportError: evaluate_model` |
| `python -m passive_walker.ppo.train` | `NameError: name 'torch' is not defined` |
| `load_bc_weights` on a BC checkpoint | `KeyError: 'weight'` |
| `PerturbationManager` in "random" mode, 10 s | 0 perturbations applied |
