# Tools

Simple utility scripts for model evaluation, comparison, and visualization.

## Available Tools

### 1. evaluate_model.py

Evaluate a BC or PPO model.

```bash
# Evaluate BC model
python tools/evaluate_model.py experiments/models/bc/model.pt --type bc --episodes 10

# Evaluate PPO model
python tools/evaluate_model.py experiments/models/ppo/model.pth --type ppo --episodes 10

# Save metrics to file
python tools/evaluate_model.py model.pt --type bc --episodes 10 --out metrics.json
```

### 2. compare_models.py

Compare multiple models side by side.

```bash
# Compare BC models
python tools/compare_models.py model1.pt model2.pt model3.pt --type bc --episodes 10

# Compare PPO models
python tools/compare_models.py model1.pth model2.pth --type ppo --episodes 10

# Save comparison results
python tools/compare_models.py model1.pt model2.pt --type bc --episodes 10 --out comparison.json
```

### 3. visualize_results.py

Plot training curves and visualization results.

```bash
# Plot training curves from TensorBoard logs
python tools/visualize_results.py experiments/runs/ppo/experiment_name --type curves --out training_curves.png

# Plot episode trajectory
python tools/visualize_results.py episode_data.json --type trajectory --out trajectory.png
```

## Usage Examples

### Evaluate a trained BC model

```bash
python tools/evaluate_model.py experiments/models/bc/torch_hip_seed123.pt --type bc --episodes 20
```

### Compare PPO model variants

```bash
python tools/compare_models.py \
    experiments/models/ppo/simple/model1.pth \
    experiments/models/ppo/simple/model2.pth \
    --type ppo --episodes 10
```

### Visualize training progress

```bash
python tools/visualize_results.py experiments/runs/ppo/my_experiment --type curves
```

## Archived Tools

Complex analysis and evaluation tools have been moved to `_archive/tools/`:
- `_archive/tools/analysis/` - Advanced analysis tools
- `_archive/tools/evaluation/` - Complex evaluation pipelines
