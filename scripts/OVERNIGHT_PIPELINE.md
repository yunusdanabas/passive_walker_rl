# Overnight Pipeline Documentation

## Overview

The `run_overnight.sh` script automates the complete training pipeline for the passive walker project. It collects FSM data, trains multiple BC and PPO models, and generates comprehensive evaluation visualizations.

## Pipeline Steps

### 1. FSM Data Collection
- Collects 100 episodes of FSM data
- Each episode: 25 seconds duration
- Output: `fsm_data/episode_*.npz` files
- Visualizations: FSM state transitions, joint angles, rewards

### 2. BC Model Training (Sweep)
Trains 9 BC models:
- **Sections**: hip only, knees only, both joints
- **Seeds**: 123, 456, 789
- **Epochs**: 50 per model
- **Output**: `bc_models/*.pt` and `bc_models/*.json`

### 3. BC Model Evaluation & Comparison
- Evaluates all 9 models with 10 episodes each
- Creates comparison plots:
  - Average return by section
  - Success rate by section
  - Individual model performance
  - Episode length distributions
- Output: `bc_evaluation/bc_models_comparison.png`

### 4. PPO Model Training (Sweep)
Trains 3 PPO models:
- **Seeds**: 42, 123, 456
- **Timesteps**: 500,000 each
- **Output**: `ppo_models/ppo_overnight_seed*/`

### 5. PPO Model Evaluation & Comparison
- Generates training metrics plots for each seed
- Creates comparison visualizations
- Output: `ppo_evaluation/` directory with per-seed plots

## Running the Pipeline

```bash
cd /home/yunusdanabas/passive_walker_rl
./scripts/run_overnight.sh
```

## Output Structure

```
experiments/overnight_run_TIMESTAMP/
├── fsm_data_visualization.png
├── fsm_data/
│   ├── episode_000000.npz
│   └── ...
├── bc_models/
│   ├── torch_hip_seed123_ep*.pt
│   ├── torch_hip_seed456_ep*.pt
│   ├── torch_hip_seed789_ep*.pt
│   ├── torch_knees_seed123_ep*.pt
│   ├── torch_knees_seed456_ep*.pt
│   ├── torch_knees_seed789_ep*.pt
│   ├── torch_both_seed123_ep*.pt
│   ├── torch_both_seed456_ep*.pt
│   └── torch_both_seed789_ep*.pt
├── bc_evaluation/
│   ├── bc_models_comparison.png
│   └── bc_summary.json
├── ppo_models/
│   ├── ppo_overnight_seed42/
│   ├── ppo_overnight_seed123/
│   └── ppo_overnight_seed456/
└── ppo_evaluation/
    ├── seed42/
    ├── seed123/
    ├── seed456/
    └── ppo_models_comparison.png
```

## Key Features

1. **Sweep Training**: Automatically trains multiple models with different configurations
2. **Automatic Comparison**: Generates comparison plots across all models
3. **No JSON Output**: Only figures and tables (as requested)
4. **Existing Code**: Uses existing modules from the codebase
5. **Simple Execution**: Single script, no manual steps

## Estimated Runtime

- FSM Data Collection: ~30 minutes
- BC Training (9 models): ~2-3 hours
- BC Evaluation: ~30 minutes
- PPO Training (3 models): ~3-4 hours
- PPO Evaluation: ~15 minutes

**Total**: 6-8 hours

## Configuration

To modify the pipeline, edit the following variables in `scripts/run_overnight.sh`:

```bash
# FSM Data Collection
EPISODES=100
DURATION=25.0

# BC Sweep
SEEDS=(123 456 789)
SECTIONS=("hip" "knees" "both")
EPOCHS=50

# PPO Sweep
PPO_SEEDS=(42 123 456)
TIMESTEPS=500000
```

## Notes

- All operations use `mamba run -n main`
- GPU is enabled for BC training if available
- Visualizations use matplotlib with publication-ready settings
- Results are saved to timestamped directories for tracking

