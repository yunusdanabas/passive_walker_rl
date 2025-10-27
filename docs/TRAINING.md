# Training Guide

This guide covers training BC and PPO models for the passive walker.

## Data Collection

### FSM Demonstration Collection

Collect demonstration data for Behavior Cloning:

```bash
# Basic collection (10 episodes, 20 seconds each)
python -m passive_walker.fsm.collect \
    --episodes 10 \
    --duration 20 \
    --out experiments/data/fsm_demos

# With observation noise
python -m passive_walker.fsm.collect \
    --episodes 10 \
    --duration 20 \
    --obs-noise 0.5 \
    --out experiments/data/noisy_demos

# With domain randomization
python -m passive_walker.fsm.collect \
    --episodes 20 \
    --duration 30 \
    --ramp-jitter 2.0 \
    --out experiments/data/robust_demos
```

**Parameters:**
- `--episodes`: Number of episodes
- `--duration`: Episode duration in seconds
- `--out`: Output directory
- `--obs-noise`: Observation noise level
- `--ramp-jitter`: Ramp angle variation

## Behavior Cloning (BC)

### Basic Training

```bash
# Train BC model
python -m passive_walker.bc.training.train \
    --data experiments/data/fsm_demos \
    --out experiments/models/bc/my_model \
    --epochs 100 \
    --seed 123
```

### Advanced Options

```bash
# With data augmentation
python -m passive_walker.bc.training.train \
    --data experiments/data/fsm_demos \
    --out experiments/models/bc/augmented_model \
    --epochs 200 \
    --augmentation temporal \
    --hidden-sizes 128 128

# Temporal model
python -m passive_walker.bc.training.train \
    --data experiments/data/fsm_demos \
    --model-type lstm \
    --hidden-size 128 \
    --out experiments/models/bc/lstm_model
```

**Output:**
- Model: `experiments/models/bc/{model_name}.pt`
- Metrics: `experiments/models/bc/{model_name}_metrics.json`
- Metadata: `experiments/models/bc/{model_name}_meta.json`

### Evaluation

```bash
# Evaluate model
python -m passive_walker.bc.evaluation.evaluate \
    --model experiments/models/bc/my_model.pt \
    --episodes 10

# Play model in GUI
python -m passive_walker.bc.evaluation.play \
    --model experiments/models/bc/my_model.pt \
    --episodes 3 \
    --gui
```

## PPO Training

### Basic Training

```bash
# Train PPO model
python -m passive_walker.ppo.train \
    --experiment_name my_ppo \
    --timesteps 100000 \
    --seed 42
```

### Model Configurations

```bash
# MLP model (default)
python -m passive_walker.ppo.train \
    --experiment_name mlp_ppo \
    --model_type mlp \
    --hidden_sizes 64 64 \
    --timesteps 200000

# LSTM model
python -m passive_walker.ppo.train \
    --experiment_name lstm_ppo \
    --model_type lstm \
    --hidden_size 128 \
    --num_layers 2 \
    --timesteps 300000

# GRU model
python -m passive_walker.ppo.train \
    --experiment_name gru_ppo \
    --model_type gru \
    --hidden_size 128 \
    --timesteps 300000
```

### Training Options

```bash
# Custom training configuration
python -m passive_walker.ppo.train \
    --experiment_name custom_ppo \
    --timesteps 500000 \
    --learning_rate 3e-4 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_epsilon 0.2 \
    --value_coef 0.5 \
    --entropy_coef 0.01
```

**Output:**
- Model: `experiments/models/ppo/{experiment_name}/final_model.pth`
- Logs: `experiments/runs/ppo/{experiment_name}/events.out.tfevents.*`

### Evaluation

```bash
# Evaluate PPO model
python -m passive_walker.ppo.evaluate \
    --model experiments/models/ppo/my_ppo/final_model.pth \
    --episodes 10

# Play in GUI
python -m passive_walker.ppo.play_ppo_model \
    --model experiments/models/ppo/my_ppo/final_model.pth \
    --episodes 3 \
    --gui
```

### Plotting Results

```bash
# Plot training curves
python -m passive_walker.ppo.plot_ppo_results \
    --logdir experiments/runs/ppo/my_ppo \
    --output experiments/analysis/plots/ppo/training_curves.png
```

## Model Locations

After reorganization:
- **BC models**: `experiments/models/bc/`
- **PPO models**: `experiments/models/ppo/`
- **Training logs**: `experiments/runs/{bc|ppo}/`
- **Analysis metrics**: `experiments/analysis/metrics/{bc|ppo}/`
- **Plots**: `experiments/analysis/plots/{bc|ppo}/`
- **Reports**: `experiments/analysis/reports/`
- **Data**: `experiments/data/fsm_demos/`

## Tips

1. **Start with BC**: Train BC first for a baseline
2. **Use appropriate data**: Collect diverse demonstrations
3. **Monitor training**: Use TensorBoard or built-in logging
4. **Validate models**: Test on multiple conditions
5. **Iterate**: Adjust hyperparameters based on results

## Next Steps

- See [API.md](API.md) for detailed API reference
- Check training scripts in `scripts/` directory
- Review model performance in `experiments/` directory

