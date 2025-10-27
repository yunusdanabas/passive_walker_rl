# Passive Walker RL Project

A reinforcement learning project for training a passive bipedal walker using Behavior Cloning (BC) and Proximal Policy Optimization (PPO).

## Overview

The passive walker is a Variable Length Leg (VLL) bipedal robot that uses an underactuated mechanism where only the hip and knee motors provide control. The project implements:

- **Finite State Machine (FSM) Controller**: Deterministic reference controller for data collection
- **Behavior Cloning (BC)**: Neural networks trained to mimic FSM behavior
- **Proximal Policy Optimization (PPO)**: Reinforcement learning for policy optimization

## Project Structure

```
passive_walker_rl/
├── passive_walker/          # Main package
│   ├── core/                # Environment and physics
│   ├── fsm/                 # FSM controller
│   ├── bc/                  # Behavior Cloning
│   └── ppo/                 # PPO training
├── experiments/             # All experiments
│   ├── models/              # Trained models
│   ├── data/                # Training data
│   └── runs/                # Training logs
├── tools/                   # Utility scripts
├── tests/                   # Unit tests
├── scripts/                 # Training scripts
└── docs/                    # Documentation
```

## Quick Start

### Installation

```bash
# Clone the repository
cd /home/yunusdanabas/passive_walker_rl

# Activate environment
mamba activate main

# Install package
pip install -e .
```

### Basic Usage

**Run FSM controller:**
```bash
python -m passive_walker.core.env --mode fsm --gui
```

**Collect demonstration data:**
```bash
python -m passive_walker.fsm.collect --episodes 10 --duration 20 --out experiments/data/fsm_demos
```

**Train BC model:**
```bash
python -m passive_walker.bc.training.train --data experiments/data/fsm_demos --out experiments/models/bc/my_model
```

**Train PPO model:**
```bash
python -m passive_walker.ppo.train --experiment_name my_ppo --timesteps 100000
```

**Play a trained model:**
```bash
# BC model
python -m passive_walker.bc.evaluation.play --model experiments/models/bc/my_model.pt --episodes 3

# PPO model
python -m passive_walker.ppo.play_ppo_model --model experiments/models/ppo/final_model.pth --episodes 3
```

## Documentation

- [Setup](SETUP.md) - Installation and dependencies
- [Training](TRAINING.md) - BC and PPO training guides
- [API](API.md) - API reference
- [Changelog](CHANGELOG.md) - Project history

## Key Features

### Environment
- Physics randomization (ramp angle, friction, mass)
- Configurable control frequency (100-200 Hz)
- Observation noise injection
- Advanced randomization profiles

### Training
- Research mode rewards (7-component system)
- Learning rate scheduling
- Data augmentation
- Structured configuration

### Evaluation
- Multi-condition testing
- Robustness analysis
- Automated reporting
- Visualization tools

## Status

Current version: 2.1.0
- Phase 1: Data Quality & Environment - Complete
- Phase 2: Training Infrastructure - Complete
- Phase 3: Evaluation & Analysis - Complete

## License

MIT License
