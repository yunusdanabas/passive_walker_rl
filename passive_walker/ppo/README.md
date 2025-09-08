# Proximal Policy Optimization (PPO)

This module handles on-policy RL training using PPO algorithm.

## Overview

PPO is a policy gradient method that learns to maximize cumulative reward through interaction with the environment. It uses clipped objective and GAE for stable learning.

## Usage

### Train PPO Policy

```bash
# Basic training
walker.train_ppo --config passive_walker/ppo/ppo_train.yaml

# Specify output directory
walker.train_ppo --output-dir results/ppo/my_experiment

# Use different device
walker.train_ppo --device cuda

# Set random seed
walker.train_ppo --seed 42
```

### Configuration

The PPO training uses `passive_walker/ppo/ppo_train.yaml` by default. Key parameters:

- `ppo.num_epochs`: Total training epochs (default: 1000)
- `ppo.max_steps_per_epoch`: Max steps per epoch (default: 1000)
- `ppo.update_epochs`: PPO updates per epoch (default: 4)
- `ppo.batch_size`: Batch size for updates (default: 64)
- `ppo.gamma`: Discount factor (default: 0.99)
- `ppo.lam`: GAE lambda (default: 0.95)
- `ppo.clip_ratio`: PPO clip ratio (default: 0.2)
- `ppo.learning_rate`: Learning rate (default: 0.0003)

### Output

Training results are saved to `results/ppo/ppo_run_YYYYmmdd-HHMMSS/` with:

- `final_policy.pt`: Final trained model
- `policy_epoch_XXX.pt`: Checkpoints every 100 epochs
- `training_metrics.csv`: Training metrics per epoch
- `episode_metrics.csv`: Episode rewards and lengths
- `meta.json`: Training metadata

## Model Architecture

The PPO policy uses an actor-critic network:

```
Input (11) -> Hidden (64) -> ReLU -> Hidden (64) -> ReLU -> [Policy Head, Value Head]
```

- **Policy Head**: Outputs action mean and log_std for each joint
- **Value Head**: Outputs state value estimate
- **Actions**: Sampled from normal distribution with learned mean/std

## Training Process

1. **Rollout Collection**: Collect episodes using current policy
2. **Advantage Estimation**: Compute GAE advantages and returns
3. **PPO Updates**: Multiple updates with clipped objective
4. **Logging**: Track training metrics and episode performance
5. **Checkpointing**: Save models periodically

## PPO Algorithm

1. **Policy Gradient**: Estimate policy gradient from collected data
2. **Clipped Objective**: Prevent large policy updates
3. **Value Function**: Learn state value for advantage estimation
4. **Entropy Bonus**: Encourage exploration

## Domain Randomization

PPO training includes physics domain randomization:

- **Ramp Angle**: Random between 8-12 degrees
- **Friction**: Random between 0.6-1.0
- **Mass**: Random torso mass ±10%

## Monitoring

Training progress is logged every 10 epochs with:
- Episode reward and average reward
- Episode length and average length
- Policy loss, value loss, and total loss
- Approximate KL divergence and clip fraction

## Troubleshooting

- **No learning**: Check learning rate and reward scaling
- **Unstable training**: Reduce learning rate or increase batch size
- **Poor exploration**: Increase entropy coefficient
- **Memory issues**: Reduce batch size or max steps per epoch
- **Slow training**: Use GPU if available

