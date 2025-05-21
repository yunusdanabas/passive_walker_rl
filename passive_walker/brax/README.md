# Brax Integration for Passive Walker

This directory contains the Brax physics engine integration for the Passive Walker project. Brax is a differentiable physics engine that enables efficient simulation and reinforcement learning for rigid body dynamics.

## Contents

- `__init__.py` - Module initialization and path configurations
- `convert_xml.py` - Utility to convert MuJoCo XML models to Brax System format
- `utils.py` - Helper functions for Brax system manipulation and analysis
- `sweep_ppo.py` - PPO (Proximal Policy Optimization) hyperparameter sweep implementation
- `tiny_ppo_sanity.py` - Lightweight PPO implementation for testing
- `benchmarks.py` - Performance benchmarking utilities
- `Brax.ipynb` - Jupyter notebook with examples and analysis

## Directory Structure

The module creates and manages the following directory structure:

```
project_root/
├─ data/
│  └─ brax/                 # Brax-specific data directory
│     └─ system.pkl.gz      # Cached Brax System (auto-generated)
└─ passiveWalker_model.xml  # Source MuJoCo XML model
```

## Usage

### Converting MuJoCo XML to Brax System

To convert a MuJoCo XML model to Brax System format:

```bash
python -m passive_walker.brax.convert_xml --xml passiveWalker_model.xml
```

### Importing in Python

```python
from passive_walker.brax import XML_PATH, SYSTEM_PICKLE, set_device
```

## Key Features

- MuJoCo XML to Brax System conversion
- PPO implementation for reinforcement learning
- Vectorized environment rollouts
- Performance benchmarking tools
- GPU acceleration support

## Dependencies

- Brax
- MuJoCo
- JAX
- NumPy

## Version

Current version: 0.1.0 