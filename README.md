# Passive Walker RL Project

A comprehensive reinforcement learning project for training a passive walker using Behavior Cloning (BC) and Proximal Policy Optimization (PPO).

## Project Structure

```
passive_walker_rl/
├── passive_walker/         # Main package
│   ├── core/              # Core environment and physics
│   ├── bc/                # Behavior Cloning
│   │   ├── models/        # Model architectures
│   │   ├── training/      # Training scripts
│   │   ├── evaluation/    # Evaluation and play scripts
│   │   └── data/          # Data loading and augmentation
│   ├── fsm/               # Finite State Machine controller
│   └── ppo/               # PPO implementation
├── experiments/            # Unified experiment outputs
│   ├── models/            # ALL trained models
│   │   ├── bc/            # BC models
│   │   └── ppo/           # PPO models
│   ├── data/              # Training datasets
│   │   └── fsm_demos/     # FSM demonstrations
│   └── runs/               # Training logs
│       ├── bc/            # BC training logs
│       └── ppo/           # PPO training logs
│   └── analysis/          # Unified analysis outputs
│       ├── metrics/       # Evaluation metrics (JSON)
│       ├── plots/         # Visualizations (bc/ppo)
│       └── reports/       # Reports (md/html)
├── tools/                  # Simple utility scripts
│   ├── evaluate_model.py  # Universal model evaluation
│   ├── compare_models.py  # Model comparison
│   └── visualize_results.py # Plotting tools
├── docs/                   # Essential documentation
│   ├── README.md          # Project overview
│   ├── SETUP.md           # Installation guide
│   ├── TRAINING.md        # Training guide
│   ├── API.md             # API reference
│   └── CHANGELOG.md       # Project history
├── tests/                  # Core tests (~6 files)
├── scripts/                # Simple training scripts
│   ├── train_bc.sh        # Train BC model
│   ├── train_ppo.sh       # Train PPO model
│   ├── collect_data.sh    # Collect FSM data
│   └── check_training_progress.sh
└── _archive/               # Archived code (reference only)
    ├── legacy/             # Original legacy code
    ├── bc/                 # Archived BC features
    ├── ppo/                # Archived PPO trainers
    ├── tools/              # Archived complex tools
    ├── tests/              # Archived tests
    └── scripts/            # Archived scripts
```

## Quick Start

### 1. Setup Environment
```bash
mamba activate main
cd /home/yunusdanabas/passive_walker_rl
pip install -e .
```

### 2. Collect FSM Demonstrations
```bash
./scripts/collect_data.sh 10 20 experiments/data/fsm_demos
```

### 3. Train BC Model
```bash
./scripts/train_bc.sh experiments/data/fsm_demos bc_v1 100
```

### 4. Train PPO Model
```bash
./scripts/train_ppo.sh
```

### 5. Evaluate Models
```bash
# Evaluate BC model
python tools/evaluate_model.py experiments/models/bc/bc_v1.pt --type bc

# Evaluate PPO model
python tools/evaluate_model.py experiments/models/ppo/final_model.pth --type ppo

# Migrate legacy outputs to unified structure (dry run)
python tools/migrate_experiments.py --dry-run
```

## Key Features

### Enhanced Environment
- **Physics Randomization**: Ramp angle, friction, mass variations
- **Configurable Control Frequency**: 100Hz, 150Hz, 200Hz
- **Observation Noise**: Gaussian noise injection
- **Advanced Randomization Profiles**: Basic, moderate, aggressive, temporal

### Enhanced Training
- **Research Mode Rewards**: 7-component reward system
- **Configuration Validation**: Structured dataclass configs
- **Learning Rate Scheduling**: Plateau, cosine, warmup schedulers
- **Data Augmentation**: Observation/action noise, temporal shifts

### Comprehensive Evaluation
- **Multi-Condition Testing**: 6+ physics conditions
- **Robustness Analysis**: Control frequency, physics variations
- **Visualization Tools**: Trajectory plots, reward analysis
- **Automated Reports**: Detailed markdown reports

## Model Performance

| Model | Success Rate | Training Data | Specialization |
|-------|-------------|---------------|----------------|
| Baseline | 100% | 80 episodes, 180k steps | Original FSM |
| Enhanced | 100% | 10 episodes, 18k steps | Diverse physics |
| Gentle | 100% | Gentle slopes | Gentle terrain |
| Low Friction | 100% | Low friction | Slippery surfaces |
| Mass Jitter | 100% | Mass randomization | Mass variations |

## Documentation

- [docs/README.md](docs/README.md) - Project overview and quick start
- [docs/SETUP.md](docs/SETUP.md) - Installation and dependencies
- [docs/TRAINING.md](docs/TRAINING.md) - BC and PPO training guides
- [docs/API.md](docs/API.md) - Core API reference
- [docs/CHANGELOG.md](docs/CHANGELOG.md) - Project history

## Key Features

- **FSM Controller**: Deterministic reference controller for data collection
- **Behavior Cloning**: Neural networks trained to mimic FSM behavior
- **PPO Training**: Simple and effective reinforcement learning
- **Unified Structure**: All experiments in single `experiments/` directory
- **Simple Tools**: Essential evaluation and comparison utilities

## Development Status

✅ **Completed**: Documentation consolidation (25+ → 5 files)  
✅ **Completed**: Experiments unified (6+ locations → single `experiments/`)  
✅ **Completed**: Code simplified (BC, PPO, tools, tests, scripts)  
✅ **Completed**: Archive organized (`_legacy/` → `_archive/legacy/`)

**Ready for training and development**
