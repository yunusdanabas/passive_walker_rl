# Passive Walker RL Environment

A unified, high-performance reinforcement learning environment for a Variable Length Leg (VLL) bipedal walker using MuJoCo physics simulation.

## Features

- **Unified Environment**: Single `PassiveWalkerEnv` with two modes (FSM data collection, RL training)
- **Reward System**: Configurable reward presets (minimal, default, aggressive) with smooth mathematical terms
- **JAX Acceleration**: Optional JIT-compiled utilities for PD control, quaternion conversion, and batched rewards
- **Memory Pooling**: Efficient rollout buffers with streaming normalization and NPZ serialization
- **Complete Workflow**: FSM data collection → BC training → PPO training

## Quick Start

### 1. Collect FSM Expert Data

```bash
python passive_walker/scripts/collect_fsm_data.py \
  --config passive_walker/configs/fsm_collect.yaml \
  --num_episodes 100 \
  --rollout_len 1000 \
  --output_dir data/bc/raw
```

### 2. Train BC Policy

```bash
python passive_walker/scripts/train_bc.py \
  --data_dir data/bc/raw \
  --out_dir results/bc \
  --epochs 100 \
  --lr 3e-4 \
  --loss huber \
  --normalize_obs
```

### 3. Train PPO Agent

```bash
python passive_walker/scripts/train_ppo.py \
  --config passive_walker/configs/ppo_train.yaml \
  --bc_init results/bc/policy.pt \
  --num_envs 8 \
  --total_steps 1000000 \
  --out_dir results/ppo
```

## Configuration

The environment is configured via YAML files in `passive_walker/configs/`:

### Core Settings

- **`mode`**: `"fsm"` (data collection) or `"research"` (RL training)
- **`env`**: Simulation parameters (timestep, control frequency, XML path)
- **`physics`**: Ramp angle, friction, mass jitter, termination thresholds
- **`control`**: PD gains, joint limits, NN control flags
- **`reward`**: Preset selection and parameter overrides
- **`jax`**: Optional JAX acceleration settings

### Reward Presets

- **`minimal`**: Forward progress only (`reward = dx`)
- **`default`**: Balanced reward with upright, velocity tracking, symmetry, foot clearance
- **`aggressive`**: High-gain reward for faster learning

### Example Config

```yaml
mode: "research"
env:
  simend: 30.0
  ctrl_hz: 60
  xml_path: "passive_walker/assets/passiveWalker_model.xml"

physics:
  ramp_deg_min: 10.0
  ramp_deg_max: 14.0
  fall_z_min: 0.15
  fall_pitch_max: 1.0

control:
  use_nn_for_hip: true
  use_nn_for_knees: true

reward:
  preset: "default"
  overrides: { c_fp: 2.0, c_up: 1.0 }

jax:
  enable: false
  batched: false
```

## Architecture

```
passive_walker/
├── core/                    # Core modules
│   ├── env.py              # Unified environment
│   ├── reward.py           # Reward system
│   ├── controller.py       # PD control + FSM
│   ├── jax_utils.py        # JAX acceleration
│   ├── rollout_buffer.py   # Memory pooling
│   ├── config.py           # Dataclasses
│   └── io.py               # YAML loading
├── configs/                 # Configuration files
│   ├── walker.yaml         # Main config
│   ├── fsm_collect.yaml    # FSM data collection
│   ├── bc_eval.yaml        # BC evaluation
│   └── ppo_train.yaml      # PPO training
├── scripts/                 # Training scripts
│   ├── collect_fsm_data.py # FSM data collection
│   ├── train_bc.py         # BC training
│   └── train_ppo.py        # PPO training
└── assets/                  # MuJoCo models
    └── passiveWalker_model.xml
```

## Key Components

### Environment (`core/env.py`)

- **Single class**: `PassiveWalkerEnv` with mode switching
- **Action space**: 3D normalized actions `[-1, 1]` for hip and knee joints
- **Observation space**: 11D state vector `[x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]`
- **Two modes**: FSM (data collection) and research (RL training)

### Reward System (`core/reward.py`)

- **Preset-based**: Three reward configurations with parameter overrides
- **Smooth terms**: Upright, velocity tracking, symmetry, foot clearance
- **Fall handling**: Termination penalties and clipping
- **Extras**: Detailed reward breakdown for analysis

### Rollout Buffer (`core/rollout_buffer.py`)

- **Memory efficient**: Preallocated arrays prevent per-step allocations
- **Streaming normalization**: Real-time observation statistics
- **Serialization**: Complete save/load with metadata and normalization stats
- **Multi-environment**: Support for vectorized PPO training

### JAX Utils (`core/jax_utils.py`)

- **PD control**: JIT-compiled proportional-derivative control
- **Quaternion conversion**: Efficient quat to Euler conversion
- **Batched rewards**: Vectorized reward computation for multiple environments

## Training Workflows

### FSM Data Collection

1. **Purpose**: Generate expert trajectories for imitation learning
2. **Mode**: `fsm` with minimal reward
3. **Control**: FSM state machine (no neural networks)
4. **Output**: NPZ files with obs, actions, rewards, and extras

### Behavioral Cloning

1. **Purpose**: Learn to imitate FSM expert behavior
2. **Input**: NPZ files from FSM collection
3. **Architecture**: Simple MLP policy
4. **Loss**: MSE, Huber, or L1 loss options
5. **Output**: Trained policy weights

### PPO Training

1. **Purpose**: Reinforcement learning with shaped rewards
2. **Mode**: `research` with configurable reward preset
3. **Architecture**: Actor-critic with shared feature extractor
4. **Initialization**: Optional BC policy weights
5. **Output**: Trained RL agent

## Performance

- **Environment**: 200+ FPS on modern hardware
- **Memory**: Zero-copy data collection with preallocated buffers
- **JAX**: Optional 2-3x speedup for batched operations
- **Serialization**: Fast NPZ format with metadata preservation

## Installation

```bash
pip install -r requirements-lock.txt
```

## Legacy Code

The `_legacy/` directory contains the original codebase for reference. **Do not import from `_legacy/`** - use the new unified core modules instead.

## Development

```bash
# Linting
ruff check passive_walker/

# Formatting
black passive_walker/

# Run tests
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.