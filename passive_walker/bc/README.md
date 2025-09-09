# Behavior Cloning (BC) — Scaffold

This folder hosts the BC pipeline (Torch and JAX backends) to mimic the FSM controller.

**Status:** Scaffold only (CLIs, structure, and conventions). Training/eval will be added next.

## Layout
- `dataset.py` — discover NPZ demos, schema checks, episode splits
- `models_torch.py` / `models_jax.py` — small MLPs with Tanh head
- `train.py` — unified CLI: `--backend {torch|jax}`, `--section {hip|knees|both|both-adv}`
- `play.py` — run saved model(s) in the env's research mode (stub)
- `utils.py` — seeds, device, json helpers
- `checkpoints/` — saved models & meta

**Backends:** Torch and JAX are fully optional; each script imports its backend lazily.