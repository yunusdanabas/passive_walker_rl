# Passive Walker RL

A comprehensive implementation of a reinforcement learning pipeline for a passive bipedal walker using JAX. This project implements a full pipeline from Finite State Machine (FSM) expert demonstrations to Brax-scaled Proximal Policy Optimization (PPO) on a minimal bipedal walker.

## Project Overview

This project transforms a rule-based toy walker into a fast-learning, resilient biped through a carefully designed curriculum:

1. **Expert FSM**: Two-phase hip cycle with knee toggling generates stable downhill "fall-and-catch" steps
2. **Behavior Cloning**: Trained on ~30k expert state-action pairs using 2×64 tanh MLP
3. **BC-Seeded PPO**: Actor initialized from BC with separate critic, achieving smoother, faster, disturbance-robust gait
4. **PPO-from-Scratch**: Demonstrates the value of curriculum learning

## Features

- JAX-native implementation for high-performance computing
- Integration with Brax physics engine for efficient rigid body dynamics
- FSM-based expert demonstrations for behavioral cloning
- PPO implementation for policy learning
- MuJoCo model support with automatic conversion to Brax
- Comprehensive training and evaluation pipeline
- GPU acceleration support
- Vectorized environment rollouts for efficient training
- Parallel training capabilities (128-1024 replicas)
- Network architecture scaling from tiny to deepXL

## Technical Stack

| Layer            | Tooling                                          | Purpose                                                  |
| ---------------- | ------------------------------------------------ | -------------------------------------------------------- |
| **Dynamics**     | MuJoCo 2.3 (MJCF) → Brax MJCF loader             | High-fidelity model; JAX-native simulation for scale-out |
| **RL API**       | OpenAI Gym + custom `PassiveWalkerEnv`           | Standardise observations/actions                         |
| **Numerics**     | JAX + Equinox + Optax                            | Autograd, neural nets, optimisers, all JIT-compiled      |
| **RL Algorithm** | `brax.training.agents.ppo`                       | Parallel, device-native PPO trainer                      |
| **Utilities**    | NumPy, SciPy Rotations, tqdm, matplotlib, pickle | Data handling, logging, visualisation                    |

## Walker Model

- Torso on slide-x/slide-z & yaw; two legs with hinge hips and prismatic knees
- Position-controlled joints (hip kₚ = 5, knees kₚ = 1000)
- Virtual downhill ramp: gravity pitched 11.5°
- Episode terminates after 1,024 steps or on fall (torso < 0.15m or pitch > 0.70rad)

## Installation

The package requires Python 3.9 or 3.10. Install the package using pip:

```bash
pip install -e .
```

### Dependencies

The project relies on the following main dependencies:
- JAX - For high-performance numerical computing
- Equinox - For neural network implementation
- Optax - For optimization algorithms
- Brax - For physics simulation
- MuJoCo - For model definition and conversion
- Gym - For environment interfaces
- NumPy - For numerical operations
- SciPy - For scientific computing
- Matplotlib - For visualization
- tqdm - For progress tracking

## Project Structure

```
passive_walker_rl/
├── passive_walker/          # Main package directory
│   ├── bc/                 # Behavioral cloning components
│   ├── brax/              # Brax physics engine integration
│   │   ├── convert_xml.py # MuJoCo to Brax converter
│   │   ├── sweep_ppo.py   # PPO hyperparameter sweep
│   │   └── utils.py       # Brax utilities
│   ├── controllers/        # Controller implementations
│   ├── envs/              # Environment definitions
│   ├── ppo/               # PPO implementation
│   ├── utils/             # Utility functions
│   └── constants.py       # Project constants
├── scripts/               # Training and evaluation scripts
├── data/                  # Data storage
│   └── brax/             # Brax-specific data
│       └── system.pkl.gz  # Cached Brax System
├── results/              # Training results and logs
└── passiveWalker_model.xml # MuJoCo model definition
```

## Usage

### Converting MuJoCo Model to Brax

To convert the MuJoCo XML model to Brax System format:

```bash
python -m passive_walker.brax.convert_xml --xml passiveWalker_model.xml
```

### Training

To train the agent using PPO:

```bash
python -m passive_walker.brax.sweep_ppo
```

For a lightweight test run:

```bash
python -m passive_walker.brax.tiny_ppo_sanity
```

### Importing in Python

```python
from passive_walker.brax import XML_PATH, SYSTEM_PICKLE, set_device
```

## Performance & Scaling

The implementation leverages JAX and Brax for high-performance computing:
- GPU acceleration support
- Vectorized environment rollouts (~45µs/env/step on RTX-class GPU)
- Efficient physics simulation
- Parallel training capabilities

### Network Architecture Scaling

| Arch    | Policy   | Value     | Status     |
| ------- | -------- | --------- | ---------- |
| tiny    | 2 × 64   | 2 × 64    | ✓ baseline |
| small   | 3 × 128  | 3 × 128   | ✓          |
| medium  | 4 × 256  | 4 × 256   | ✓          |
| deepXL  | 12 × 512 | 14 × 1024 | ✓ (fits)   |
| deepXXL | 16 × 768 | 18 × 1536 | ✗ (OOM)    |

## Key Findings

1. **Curriculum > cold-start**: BC-seeded PPO converges in < 1M steps; scratch PPO lags
2. **Capacity sweet-spot**: Performance improves up to deepXL; deepXXL overwhelms GPU memory
3. **Stable training recipe**: Mild reward scale (1.0) + low LR (≤ 3e-4) prevents PPO divergence
4. **Throughput leap**: Brax delivers two orders-of-magnitude faster data collection than single-env MuJoCo

## Challenges & Solutions

| Issue              | Symptom                  | Mitigation                     |
| ------------------ | ------------------------ | ------------------------------ |
| GPU OOM on deepXXL | XLA allocator crash      | Drop XXL or shrink batch/width |
| NaN torques        | "CTRL = nan" in MuJoCo   | Action clamp + tanh rescale    |
| Slow scratch PPO   | Reward plateau           | Curriculum + longer horizon    |
| Over-specific gait | Fails on friction change | Domain randomisation plan      |

## Future Work

1. Complete 4-architecture Brax sweep & log reward, wall-clock, memory
2. Run ≥ 3 seeds on best/worst nets to quantify variance
3. Publish analysis notebook: sample efficiency curves, capacity-vs-return & memory-vs-FPS plots
4. Add robustness study: slope, mass, friction randomisation, richer rewards
5. Transfer best Brax policy back to MuJoCo and eventually to a hardware exosuit test-rig

## Author

Yunus Emre Danabaş

## Version

Current version: 0.1.0


