# Passive Walker RL Project

A comprehensive reinforcement learning project for training a passive walker using Behavior Cloning (BC) and Proximal Policy Optimization (PPO).

## Project Structure

```
passive_walker_rl/
├── passive_walker/          # Main package
│   ├── core/               # Core environment and physics
│   ├── bc/                 # Behavior Cloning implementation
│   ├── fsm/                # Finite State Machine controller
│   └── ppo/                # PPO implementation
├── checkpoints/             # Trained model checkpoints
│   ├── checkpoints_baseline/
│   ├── checkpoints_enhanced/
│   ├── checkpoints_gentle/
│   ├── checkpoints_low_friction/
│   └── checkpoints_mass_jitter/
├── evaluation_scripts/      # Model evaluation scripts
│   ├── comprehensive_evaluation.py
│   ├── comprehensive_comparison.py
│   ├── evaluate_proper.py
│   └── evaluate_comparison.py
├── reports/                 # Analysis reports
│   ├── FINAL_REPORT.md
│   └── TRAINING_EVALUATION_RESULTS.md
├── outputs/                 # Evaluation outputs
│   ├── evaluation_plots/
│   ├── evaluation_reports/
│   └── evaluation_results/
├── data/                    # Training data
├── tests/                   # Unit tests
├── analysis/                # Analysis tools
└── _legacy/                 # Legacy code
```

## Quick Start

### 1. Environment Setup
```bash
mamba activate main
cd /home/yunusdanabas/passive_walker_rl
```

### 2. Run Environment
```bash
# Basic FSM mode
python -m passive_walker.core.env --mode fsm --gui

# With enhanced features
python -m passive_walker.core.env --mode fsm --gui --ramp-jitter 2.0 --ctrl-hz 150
```

### 3. Collect Data
```bash
# Basic collection
python -m passive_walker.fsm.collect --episodes 10 --duration 20 --out data/demos

# With observation noise
python -m passive_walker.fsm.collect --episodes 10 --duration 20 --obs-noise 0.5 --out data/noisy_demos
```

### 4. Train Models
```bash
# Train BC model
python -m passive_walker.bc.train --data data/demos --out checkpoints/my_model
```

### 5. Evaluate Models
```bash
# Comprehensive evaluation
python evaluation_scripts/comprehensive_evaluation.py

# Quick comparison
python evaluation_scripts/evaluate_proper.py
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

- **Checkpoints**: See `checkpoints/README.md`
- **Evaluation**: See `evaluation_scripts/README.md`
- **Commands**: See `COMMANDS.md`
- **Analysis**: See `analysis/README.md`

## Status

✅ **Phase 1**: Data Quality & Environment Enhancement - Complete
✅ **Phase 2**: Training Infrastructure & Reward Shaping - Complete  
✅ **Phase 3**: Evaluation & Analysis - Complete

**Ready for PPO transition with enhanced capabilities**

## Next Steps

1. **PPO Training**: Use enhanced reward system and robustness features
2. **Model Comparison**: Evaluate PPO vs BC performance
3. **Robustness Testing**: Test on challenging physics conditions
4. **Real-world Transfer**: Validate on physical hardware
