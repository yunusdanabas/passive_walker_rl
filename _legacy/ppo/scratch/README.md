# PPO Training from Scratch

This directory contains the implementation of Proximal Policy Optimization (PPO) training from scratch for the Passive Walker environment. The implementation uses JAX and Equinox for neural network training, and MuJoCo for the physics simulation.

## Files Overview

- `run_pipeline.py`: Main script to run the end-to-end PPO training pipeline
- `train.py`: Implementation of PPO training algorithm
- `evaluate.py`: Script for evaluating trained policies
- `collect.py`: Utilities for collecting trajectory data
- `utils.py`: Helper functions and utilities
- `__init__.py`: Package initialization and constants

## Usage

### Training from Scratch

To train a PPO policy from scratch:

```bash
python -m passive_walker.ppo.scratch.run_pipeline [--gpu] [--sim-duration S] [--hz HZ]
```

Options:
- `--gpu`: Use GPU acceleration (optional)
- `--sim-duration`: Duration of the final GUI rollout in seconds (default: 30.0)
- `--hz`: Simulation frequency in Hz (default: specified in constants)

### Training Parameters

The training process can be customized with various hyperparameters:

- Number of PPO iterations
- Rollout steps per iteration
- PPO epochs per update
- Batch size
- Discount factor (gamma)
- GAE lambda
- PPO clipping epsilon
- Policy standard deviation
- Learning rates for policy and critic networks

## Implementation Details

The implementation includes:

1. A policy network for action selection
2. A critic network for value estimation
3. PPO algorithm with:
   - Generalized Advantage Estimation (GAE)
   - Clipped objective function
   - Separate policy and value function optimization
4. Training visualization and logging
5. Model checkpointing and saving

## Output

The training process generates:
- Trained policy and critic models
- Training logs with reward history
- Training curve visualization
- Demo rollout in MuJoCo GUI

## Dependencies

- JAX
- Equinox
- MuJoCo
- NumPy
- Optax (for optimization)

## Notes

- The implementation is designed to work with the Passive Walker environment
- Training can be performed on both CPU and GPU
- The code includes utilities for visualization and evaluation of trained policies 