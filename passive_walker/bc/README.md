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
| Combined Loss | Composite loss function | 3 | MSE + L1 + Huber |

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
# Collect demonstration data
python -m passive_walker.bc.hip_knee_alternatives.collect [options]

# Train with different loss functions
python -m passive_walker.bc.hip_knee_alternatives.train_mse [options]
python -m passive_walker.bc.hip_knee_alternatives.train_huber [options]
python -m passive_walker.bc.hip_knee_alternatives.train_l1 [options]
python -m passive_walker.bc.hip_knee_alternatives.train_combined [options]

# Run comparison pipeline
python -m passive_walker.bc.hip_knee_alternatives.run_comparison_pipeline [options]
```

Options:
- `--steps N`: Number of demonstration steps (default: 20,000)
- `--hz H`: Simulation frequency in Hz (default: 200)
- `--gpu`: Use GPU if available
- `--sim-duration S`: Test simulation duration in seconds (default: 15.0)

### File Naming Convention

The implementation uses a consistent file naming convention based on the number of steps:
- Demo files: `hip_knee_alternatives_demos_{steps}steps.pkl`
- Model files: `hip_knee_alternatives_{loss_type}_{steps}steps.npz`

### Interactive Development

For interactive development and exploration, use the `BehaviourClonning.ipynb` notebook. The notebook provides a step-by-step walkthrough of the behavior cloning process with visualizations and explanations.

## Implementation Details

### Data Collection
- Uses the FSM controller to generate demonstration data
- Collects state observations and corresponding actions
- Supports configurable number of steps and simulation frequency
- Saves data with step count in filename

### Training
- Implements behavior cloning using JAX and Equinox
- Supports various loss functions:
  - MSE (Mean Squared Error)
  - Huber (Robust to outliers)
  - L1 (Absolute Error)
  - Combined (MSE + L1 + Huber)
- Uses shared training infrastructure
- Saves models with step count in filename

### Evaluation
- Visualizes training progress and results
- Provides GUI-based testing environment
- Supports comparison between different variants
- Automatically selects best performing model

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
- Consistent file naming based on step count
- Shared training infrastructure across variants 