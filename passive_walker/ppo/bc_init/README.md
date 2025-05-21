# BC-Seeded PPO for Passive Walker

This module implements Proximal Policy Optimization (PPO) with Behavior Cloning (BC) initialization for training robust walking policies in the Passive Walker environment using MuJoCo.

## Overview

BC-seeded PPO combines the benefits of imitation learning and reinforcement learning:

1. **Behavior Cloning Initialization**: Start with a pre-trained policy learned from demonstrations
2. **Proximal Policy Optimization**: Refine the policy through reinforcement learning while maintaining proximity to the original behavior
3. **Annealed BC Regularization**: Gradually decrease imitation influence to allow policy improvement

This approach helps solve exploration challenges in complex locomotion tasks by starting from a reasonable initial policy.

## Requirements

- Python 3.7+
- JAX
- Equinox
- MuJoCo
- NumPy
- Matplotlib (for visualization)

## Directory Structure

```
passive_walker/ppo/bc_init/
├── __init__.py           # Module configuration and path setup
├── utils.py              # Utility functions for policy handling and RL algorithms
├── collect.py            # Script to collect trajectory data
├── train.py              # Main PPO training implementation
├── evaluate.py           # Policy evaluation and visualization
├── run_pipeline.py       # End-to-end pipeline script
```

## Usage

### Complete Pipeline

To run the entire BC-PPO training and demonstration pipeline:

```bash
python -m passive_walker.ppo.bc_init.run_pipeline \
    --bc-model /path/to/bc_model.pkl \
    --sim-duration 30.0 \
    [--gpu]
```

This will:
1. Train a PPO agent using the BC model as initialization
2. Save the trained policy with critic
3. Demonstrate the policy in the environment

### Individual Components

#### Training

```bash
python -m passive_walker.ppo.bc_init.train \
    --bc-model /path/to/bc_model.pkl \
    --iters 200 \
    --rollout 2048 \
    --bc-coef 1.0 \
    --anneal 200000 \
    [--gpu]
```

#### Data Collection

```bash
python -m passive_walker.ppo.bc_init.collect \
    --bc-model /path/to/bc_model.pkl \
    --steps 4096 \
    --sigma 0.1 \
    [--gpu]
```

#### Evaluation

```bash
python -m passive_walker.ppo.bc_init.evaluate \
    --policy /path/to/trained_policy_with_critic.pkl \
    --sim-duration 30.0 \
    [--gpu]
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--bc-model` | Path to pretrained Behavior Cloning model (.pkl) |
| `--iters` | Number of PPO training iterations |
| `--rollout` | Number of environment steps per iteration |
| `--epochs` | Number of SGD epochs per iteration |
| `--batch` | Minibatch size for PPO updates |
| `--gamma` | Discount factor for rewards |
| `--lam` | GAE lambda parameter |
| `--clip` | PPO clipping parameter |
| `--sigma` | Standard deviation for exploration |
| `--lr-policy` | Learning rate for policy updates |
| `--lr-critic` | Learning rate for critic updates |
| `--bc-coef` | Behavior Cloning regularization coefficient |
| `--anneal` | Steps to anneal BC coefficient to zero |
| `--gpu` | Flag to use GPU acceleration |

## Implementation Details

### Critic Network

A 2-layer MLP value function with ReLU activations estimates state values for advantage computation.

### Policy Updates

The implementation uses:
- Clipped PPO objective for stable policy improvement
- Behavioral Cloning regularization term to maintain proximity to the demonstration policy
- Annealed BC coefficient that gradually reduces imitation influence
- Generalized Advantage Estimation (GAE) for advantange calculation

### BC Regularization 

The BC regularization coefficient is annealed according to:
```
bc_coef = initial_bc_coef * max(0.0, 1 - total_steps/anneal_steps)
```

## Data Management

All training data and policies are stored in `data/ppo/bc_init` directory.

## Citation

If you use this implementation in your research, please cite:

```
@article{SchulmanPPO2017,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}
``` 