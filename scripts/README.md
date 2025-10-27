# Scripts

This directory contains automation scripts for the passive walker project.

## Available Scripts

### `collect_data.sh`
Collects FSM data for training.

```bash
./scripts/collect_data.sh [episodes] [duration] [output_dir]
```

### `train_bc.sh`
Trains BC models on collected data.

```bash
./scripts/train_bc.sh [data_dir] [model_name] [epochs]
```

### `train_ppo.sh`
Trains PPO models.

```bash
./scripts/train_ppo.sh
```

### `run_overnight.sh`
Runs the complete overnight pipeline:
1. Collects FSM data
2. Trains BC models sweep (3 sections × 3 seeds = 9 models)
3. Evaluates and compares BC models
4. Trains PPO models sweep (3 seeds = 3 models)
5. Evaluates and compares PPO models

```bash
./scripts/run_overnight.sh
```

All results are saved to `experiments/overnight_run_TIMESTAMP/`.

## Usage Example

Run the complete overnight pipeline:

```bash
cd /home/yunusdanabas/passive_walker_rl
./scripts/run_overnight.sh
```

This will:
- Take several hours to complete
- Save all results to a timestamped directory
- Train 9 BC models (hip, knees, both × 3 seeds)
- Train 3 PPO models (3 different seeds)
- Generate comparison visualizations (no JSON output)

## Output Structure

```
experiments/overnight_run_TIMESTAMP/
├── fsm_data/                      # Collected FSM data
│   ├── episode_000000.npz
│   └── ...
├── fsm_data_visualization.png     # FSM data plots
├── bc_models/                     # Trained BC models (9 total)
│   ├── torch_hip_seed123_*.pt
│   ├── torch_hip_seed456_*.pt
│   ├── torch_hip_seed789_*.pt
│   ├── torch_knees_seed123_*.pt
│   ├── ...
│   └── torch_both_seed789_*.pt
├── bc_evaluation/                 # BC evaluation and comparison
│   ├── bc_models_comparison.png  # Comparison plots
│   └── bc_summary.json           # Summary metrics
├── ppo_models/                    # Trained PPO models (3 total)
│   ├── ppo_overnight_seed42/
│   ├── ppo_overnight_seed123/
│   └── ppo_overnight_seed456/
└── ppo_evaluation/               # PPO evaluation plots
    ├── seed42/
    ├── seed123/
    ├── seed456/
    └── ppo_models_comparison.png
```

## BC Model Sweep

The script trains BC models with:
- **Sections**: hip only, knees only, both joints
- **Seeds**: 123, 456, 789
- **Total**: 9 models

Results are compared automatically with visualizations showing:
- Average return by section
- Success rate by section
- Individual model performance
- Episode length distributions

## PPO Model Sweep

The script trains PPO models with:
- **Seeds**: 42, 123, 456
- **Timesteps**: 500,000 each
- **Total**: 3 models

Each PPO run generates:
- Training metrics plots
- Evaluation metrics plots
- Comparison plots across seeds

## Notes

- All scripts activate the `main` mamba environment automatically
- Visualizations use matplotlib (figures and tables only)
- No raw JSON output is shown to user
- Scripts use existing code modules
- BC models are evaluated with 10 episodes each
- PPO models are evaluated with automatic plotting from TensorBoard logs
- Complete pipeline takes 4-6 hours depending on GPU availability
