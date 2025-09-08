# Behavioral Cloning (BC)

This module handles training neural network policies to imitate FSM expert demonstrations.

## Overview

Behavioral cloning learns to map observations to actions by supervised learning on expert demonstration data. The policy is trained to minimize the difference between predicted and expert actions.

## Usage

### Train BC Policy

```bash
# Basic training
walker.train_bc --data-dir data/bc/raw/fsm_collection_20240101-120000

# With custom config
walker.train_bc --config passive_walker/bc/bc_train.yaml --data-dir data/bc/raw/latest

# Specify output directory
walker.train_bc --data-dir data/bc/raw/latest --output-dir results/bc/my_experiment

# Use different loss function
walker.train_bc --data-dir data/bc/raw/latest --loss-type huber
```

### Configuration

The BC training uses `passive_walker/bc/bc_train.yaml` by default. Key parameters:

- `bc.loss_type`: Loss function ("mse", "huber", "l1")
- `bc.num_epochs`: Training epochs (default: 100)
- `bc.learning_rate`: Learning rate (default: 0.001)
- `bc.batch_size`: Batch size (default: 32)
- `bc.hidden_dims`: MLP hidden layer sizes (default: [64, 64])

### Output

Training results are saved to `results/bc/bc_run_YYYYmmdd-HHMMSS/` with:

- `best_policy.pt`: Best model based on validation loss
- `final_policy.pt`: Final model after training
- `metrics.csv`: Training/validation loss per epoch
- `eval_results.json`: Quick evaluation results
- `meta.json`: Training metadata

## Model Architecture

The BC policy uses a simple MLP:

```
Input (11) -> Hidden (64) -> ReLU -> Hidden (64) -> ReLU -> Output (3) -> Tanh
```

- Input: 11-dimensional observation vector
- Output: 3-dimensional action vector (hip, left_knee, right_knee)
- Activation: Tanh to ensure actions in [-1, 1]

## Training Process

1. **Data Loading**: Load FSM episodes from NPZ files
2. **Train/Val Split**: 80/20 split for training/validation
3. **Training Loop**: Minimize loss between predicted and expert actions
4. **Evaluation**: Quick evaluation on 5 episodes
5. **Saving**: Save best and final models

## Loss Functions

- **MSE**: Mean squared error (default)
- **Huber**: Robust to outliers
- **L1**: L1 loss for sparse solutions

## Evaluation

After training, the model is evaluated on 5 episodes with:
- Average reward and standard deviation
- Average episode length and standard deviation
- Results saved to `eval_results.json`

## Troubleshooting

- **High validation loss**: Try different loss function or increase model capacity
- **Overfitting**: Reduce model size or add regularization
- **Poor performance**: Check data quality and FSM collection parameters
- **Memory issues**: Reduce batch size or use smaller model

See also: `docs/pipeline_overview.md` and `docs/datasets.md`.
