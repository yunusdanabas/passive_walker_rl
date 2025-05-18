# Behavior Cloning for Passive Walker

This module implements behavior cloning (BC) to train neural network controllers for the passive walker robot. The implementation replaces hand-crafted FSM controllers with small neural MLPs, exploring different variants and approaches.

## Overview

The behavior cloning pipeline consists of four main steps:
1. **Collect FSM demo data** - Gather demonstration data from the FSM controller
2. **Train** a small MLP via behavior cloning
3. **Visualize** label distribution, loss curves, and predictions
4. **Run** a short GUI roll-out to inspect the gait

## Variants

The implementation includes several variants:

| Variant | Description | Output Dim | Key Features |
|---------|-------------|------------|--------------|
| Hip-only (MSE) | Baseline 1-D MLP for hip control | 1 | Simple MSE loss |
| Knee-only (MSE) | Two separate 1-D MLPs for knees | 2 | Independent knee control |
| Hip + Knee (MSE) | Single 3-D MLP for all joints | 3 | Unified control |
| Alternative Losses | Various loss functions | 3 | MSE, Huber, L1 losses |
| Combined Loss | Composite loss function | 3 | MSE + symmetry + smoothness + energy |

## Directory Structure

```
bc/
├── BehaviourClonning.ipynb    # Interactive notebook for exploration
├── plotters.py                # Visualization utilities
├── hip_mse/                   # Hip-only MSE implementation
├── knee_mse/                  # Knee-only MSE implementation
├── hip_knee_mse/             # Combined hip+knee MSE
└── hip_knee_alternatives/    # Alternative loss functions
```

## Usage

### Running the Pipeline

The complete behavior cloning pipeline can be run using the command-line interface:

```bash
python -m passive_walker.bc.hip_mse.run_pipeline [options]
```

Options:
- `--steps N`: Number of demonstration steps (default: 20,000)
- `--epochs E`: Training epochs (default: 50)
- `--batch B`: Batch size (default: 32)
- `--hidden-size H`: Hidden layer size (default: 128)
- `--lr LR`: Learning rate (default: 1e-4)
- `--sim-duration S`: Test simulation duration in seconds (default: 30.0)
- `--seed SEED`: Random seed (default: 42)
- `--gpu`: Use GPU if available
- `--plot`: Plot training loss curve

### Interactive Development

For interactive development and exploration, use the `BehaviourClonning.ipynb` notebook. The notebook provides a step-by-step walkthrough of the behavior cloning process with visualizations and explanations.

## Implementation Details

### Data Collection
- Uses the FSM controller to generate demonstration data
- Collects state observations and corresponding actions
- Supports configurable number of steps and simulation parameters

### Training
- Implements behavior cloning using JAX and Equinox
- Supports various loss functions and optimization strategies
- Includes batch processing and validation

### Evaluation
- Visualizes training progress and results
- Provides GUI-based testing environment
- Supports comparison between different variants

## Dependencies

- JAX
- Equinox
- Optax
- MuJoCo
- NumPy
- Matplotlib (for visualization)

## Notes

- The implementation is designed to be modular and extensible
- Each variant can be run independently
- Results and trained models are saved for later use
- GPU acceleration is supported but optional 