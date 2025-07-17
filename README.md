# Passive Walker RL 🏃‍♂️💨  
*A curriculum-driven, JAX-native pipeline for stable bipedal walking — from finite-state experts to Brax-scaled PPO.*

---

> **TL;DR** – We start with a hand-crafted finite-state machine (FSM), distil it into a tiny MLP via behaviour cloning (BC), then fine-tune that policy with Proximal Policy Optimisation (PPO) in a massively-vectorised **Brax** simulator.  
> A single GPU runs a **120-job sweep** in minutes, producing a smooth, disturbance-robust downhill gait.

---

## Table of Contents
1. [Key Features](#key-features)  
2. [Quick Start](#quick-start)  
3. [Installation](#installation)  
4. [Repository Layout](#repository-layout)  
5. [Core Workflows](#core-workflows)  
6. [Reproducing Results](#reproducing-results)  
7. [Design Choices & Notes](#design-choices--notes)  
8. [Troubleshooting](#troubleshooting)  
9. [Contributing](#contributing)  
---

## Key Features
| Layer | What You Get | Why It Matters |
|-------|--------------|----------------|
| **FSM → BC** | Rule-based gait distilled into a 2-layer MLP (≤ 1 k params) | Instant “walk-from-day-one” initialisation |
| **PPO (BC-seeded & scratch)** | Separate actor/critic, GAE, clip objective, annealed BC term | Sample-efficient fine-tuning & clean ablations |
| **MuJoCo ⇄ Brax** | 1-line conversion script produces a frozen `System.pkl.gz` | High-fidelity design *and* 100× faster roll-outs |
| **Vectorised JAX** | 128–1024 envs in lock-step at **< 50 µs/env/step** (RTX class) | Massive hyper-parameter sweeps in minutes |
| **Extensible BC losses** | MSE / Huber / L1 / Composite variants | Study robustness vs. over-fitting |
| **Modular CLI & logs** | Every stage is a one-liner; artefacts saved with hash-rich filenames | Reproducible pipelines & easy comparison |

---

## Quick Start
```bash
# 1.  (Optional) create & activate a fresh venv
python -m venv .venv && source .venv/bin/activate

# 2.  Editable install (CPU JAX by default)
pip install -e .

# 3.  Convert the MuJoCo model to Brax (one-time, cached)
python -m passive_walker.brax.convert_xml --xml passiveWalker_model.xml

# 4.  Smoke-test: run a 10 k-step PPO sanity check (CPU or GPU)
python -m passive_walker.brax.tiny_ppo_sanity --steps 10000

# 5.  Full BC-seeded PPO pipeline on GPU
python -m passive_walker.ppo.bc_init.run_pipeline --device gpu --hz 1000

# 6.  Hyper-parameter sweep (120 jobs, RTX 4060 Ti ≈ 20 min)
python -m passive_walker.brax.sweep_ppo --device gpu
````

GUI roll-outs (`--gui`) require the MuJoCo viewer; headless runs work everywhere.

---

## Installation

### Requirements

| Package                                                                            | Version tested                     |
| ---------------------------------------------------------------------------------- | ---------------------------------- |
| Python 3.9 / 3.10                                                                  |                                    |
| **JAX**                                                                            | 0.4 + (CPU) or 0.4 + CUDA 12 wheel |
| **Brax 2 (MJX)**                                                                   | `pip install brax==0.10.*`         |
| **MuJoCo 2.3**                                                                     | `pip install mujoco`               |
| `equinox`, `optax`, `gymnasium`, `numpy`, `scipy`, `matplotlib`, `tqdm`, `pickle5` |                                    |

> 🔧 **GPU JAX:** follow the [official wheels](https://github.com/google/jax#installation) for your CUDA + cudNN stack.

### Editable install

```bash
git clone https://github.com/YOUR_USER/passive_walker_rl.git
cd passive_walker_rl
pip install -e ".[demo]"
```

The `demo` extra installs MuJoCo and matplotlib for visualisation.

---

## Repository Layout

```
passive_walker_rl/
├─ passive_walker/          ← import pkg
│  ├─ bc/                   ← behaviour-cloning variants
│  ├─ ppo/                  ← BC-seeded & scratch PPO
│  ├─ brax/                 ← MJCF→Brax + sweeping utils
│  ├─ controllers/          ← fsm/  nn/
│  ├─ envs/                 ← MuJoCo Gym wrapper
│  └─ utils/                ← IO, plotting, device helpers
├─ data/                    ← demonstrations & cached Brax System
├─ results/                 ← all logs, checkpoints, plots
└─ passiveWalker_model.xml  ← canonical MuJoCo walker model
```

---

## Core Workflows

### 1 · Generate Expert Demonstrations

```bash
python -m passive_walker.bc.hip_knee_mse.collect         \
       --steps 30000 --hz 1000 --save data/hip_knee_mse/
```

Outputs `hip_knee_demos_30k.pkl` (\~80 MB).

### 2 · Behaviour Cloning (full hip + knee)

```bash
python -m passive_walker.bc.hip_knee_mse.train           \
       --demos data/hip_knee_mse/hip_knee_demos_30k.pkl  \
       --epochs 100 --lr 1e-4
```

Creates `policy_1000hz.eqx` plus loss curves.

### 3 · BC-Seeded PPO

```bash
python -m passive_walker.ppo.bc_init.run_pipeline        \
       --init   results/bc/hip_knee_mse/policy_1000hz.eqx \
       --device gpu --total-steps 5_000_000
```

Early iterations mix PPO loss with an annealed BC penalty.

### 4 · Scratch PPO

```bash
python -m passive_walker.ppo.scratch.run_pipeline --device gpu
```

### 5 · MuJoCo ⇄ Brax & Hyper-Parameter Sweep

```bash
python -m passive_walker.brax.convert_xml   # one-time
python -m passive_walker.brax.sweep_ppo     # 120 configs
```

Results saved under `results/brax/sweeps/YYYY-mm-dd_HHMM/`.

### 6 · Visualise

```bash
python -m passive_walker.utils.walker_plotter \
       --log results/bc/hip_knee_mse/loss.pkl
```

Generates PNGs for rewards, BC coefficients, joint kinematics, etc.

---

## Reproducing Results

| Stage                    | Script                        | Args                | Wall-clock (RTX 4060 Ti) |
| ------------------------ | ----------------------------- | ------------------- | ------------------------ |
| **FSM demo**             | `bc/*/collect.py`             | `--steps 3e4`       | < 1 min                  |
| **Full BC (Huber)**      | `bc/hip_knee_mse/train.py`    | `--epochs 100`      | ≈ 2 min                  |
| **BC-PPO (best config)** | `ppo/bc_init/run_pipeline.py` | `--total-steps 5e6` | ≈ 8 min                  |
| **120-job sweep**        | `brax/sweep_ppo.py`           | default             | ≈ 20 min                 |

All checkpoints contain SHA-256 config hashes; reruns with identical flags overwrite nothing.

---

## Design Choices & Notes

* **Curriculum beats cold-start.** Seeding PPO with BC slashes sample complexity (> 10× in our ablations).
* **Reward scaling** (0.5) and **mid-range LR** (5×10⁻⁴) proved most stable in the sweep.
* **1 M-param “medium” nets** hit the sweet-spot between expressivity and memory.
* **Dual simulators.** MuJoCo for fidelity & nice GUI; Brax for brute-force data. Both stay bit-for-bit consistent on 1 k test steps.

---

## Troubleshooting

| Symptom                           | Likely Cause                 | Fix                                                                     |
| --------------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| **`nan` torques / diverging PPO** | Out-of-range actions         | Clamp via `--tanh-action` or lower LR                                   |
| **GPU OOM** (deepXXL)             | Model too wide               | Reduce `--arch deep` or batch size                                      |
| **No GPU JAX wheel**              | Mismatched CUDA toolkit      | Follow [JAX install matrix](https://github.com/google/jax#installation) |
| **MuJoCo viewer freezes**         | Remote SSH without X-forward | Use `--gui headless` or `ssh -X`                                        |

---

## Contributing

Pull-requests & issues are welcome! Please follow the standard GitHub flow:

1. Fork → feature-branch → commit + descriptive message
2. `make style` (black, isort) or `pre-commit run --all`
3. Draft PR with checklist ticked
4. CI must pass (CPU sanity, flake8) before review.

---

**Happy walking!** Questions or collaboration ideas? Open an issue or ping
`yunusdanabas [at] sabanciuniv.edu`.
