# Setup Guide

## Prerequisites

- Python >= 3.8
- Mamba or Conda
- MuJoCo >= 2.3.0

## Environment Setup

### 1. Activate Environment

```bash
mamba activate main
cd /home/yunusdanabas/passive_walker_rl
```

### 2. Install Dependencies

```bash
# Install the package in editable mode
pip install -e .
```

### 3. Verify Installation

```bash
# Test basic import
python -c "import passive_walker; print(passive_walker.__version__)"

# Test environment
python -m passive_walker.core.env --mode fsm --gui
```

## Dependencies

### Core Dependencies
- `numpy >= 1.20.0` - Numerical computing
- `mujoco >= 2.3.0` - Physics simulation
- `gymnasium >= 0.26.0` - RL environment interface
- `torch >= 1.9.0` - PyTorch for training
- `jax >= 0.3.0` - JAX for alternative backend
- `matplotlib >= 3.3.0` - Plotting
- `pytest >= 6.0.0` - Testing

### Optional Dependencies
- TensorBoard for training visualization
- Weights & Biases for experiment tracking

## Project Structure

```
passive_walker_rl/
├── passive_walker/          # Main package
│   ├── core/               # Environment
│   ├── bc/                 # Behavior Cloning
│   ├── fsm/                # FSM controller
│   └── ppo/                # PPO training
├── experiments/            # All experiments
│   ├── models/             # Trained models
│   ├── data/               # Training data
│   └── runs/               # Training logs
└── scripts/                # Utility scripts
```

## Troubleshooting

### Import Errors
```bash
# Reinstall package
pip install -e .
```

### MuJoCo Not Found
```bash
# Install MuJoCo
pip install mujoco
```

### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Configuration

Environment variables can be set in your shell:
```bash
# Set device
export CUDA_VISIBLE_DEVICES=0

# Set num threads
export OMP_NUM_THREADS=4
```

## Next Steps

- Read [TRAINING.md](TRAINING.md) to start training
- Read [API.md](API.md) for detailed API usage
- Check [CHANGELOG.md](CHANGELOG.md) for updates

