# Passive Walker RL

A comprehensive implementation of a reinforcement learning pipeline for a passive bipedal walker using JAX. This project implements a full pipeline from Finite State Machine (FSM) expert demonstrations to Brax-scaled Proximal Policy Optimization (PPO) on a minimal bipedal walker.

## Features

- JAX-native implementation for high-performance computing
- Integration with Brax physics engine for efficient rigid body dynamics
- FSM-based expert demonstrations for behavioral cloning
- PPO implementation for policy learning
- MuJoCo model support with automatic conversion to Brax
- Comprehensive training and evaluation pipeline
- GPU acceleration support
- Vectorized environment rollouts for efficient training

## Installation

The package requires Python 3.9 or 3.10. Install the package using pip:

```bash
pip install -e .
```

### Dependencies

The project relies on the following main dependencies:
- JAX - For high-performance numerical computing
- Equinox - For neural network implementation
- Optax - For optimization algorithms
- Brax - For physics simulation
- MuJoCo - For model definition and conversion
- Gym - For environment interfaces
- NumPy - For numerical operations
- SciPy - For scientific computing
- Matplotlib - For visualization
- tqdm - For progress tracking

## Project Structure

```
passive_walker_rl/
├── passive_walker/          # Main package directory
│   ├── bc/                 # Behavioral cloning components
│   ├── brax/              # Brax physics engine integration
│   │   ├── convert_xml.py # MuJoCo to Brax converter
│   │   ├── sweep_ppo.py   # PPO hyperparameter sweep
│   │   └── utils.py       # Brax utilities
│   ├── controllers/        # Controller implementations
│   ├── envs/              # Environment definitions
│   ├── ppo/               # PPO implementation
│   ├── utils/             # Utility functions
│   └── constants.py       # Project constants
├── scripts/               # Training and evaluation scripts
├── data/                  # Data storage
│   └── brax/             # Brax-specific data
│       └── system.pkl.gz  # Cached Brax System
├── results/              # Training results and logs
└── passiveWalker_model.xml # MuJoCo model definition
```

## Usage

### Converting MuJoCo Model to Brax

To convert the MuJoCo XML model to Brax System format:

```bash
python -m passive_walker.brax.convert_xml --xml passiveWalker_model.xml
```

### Training

To train the agent using PPO:

```bash
python -m passive_walker.brax.sweep_ppo
```

For a lightweight test run:

```bash
python -m passive_walker.brax.tiny_ppo_sanity
```

### Importing in Python

```python
from passive_walker.brax import XML_PATH, SYSTEM_PICKLE, set_device
```

## Model

The project uses a minimal bipedal walker model defined in `passiveWalker_model.xml`. The model is designed to be simple yet capable of demonstrating passive walking dynamics. The model is automatically converted to Brax format for efficient simulation and training.

## Performance

The implementation leverages JAX and Brax for high-performance computing:
- GPU acceleration support
- Vectorized environment rollouts
- Efficient physics simulation
- Parallel training capabilities

## Author

Yunus Emre Danabaş

## Version

Current version: 0.1.0


