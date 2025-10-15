# Behavior Cloning (BC) Pipeline

Complete BC training and evaluation system for mimicking FSM controller behavior.
Supports both PyTorch and JAX backends with unified CLI interfaces.

## Features

- **Multi-backend support**: PyTorch and JAX implementations
- **Flexible control sections**: hip-only, knees-only, or full control
- **Advanced loss functions**: L1, MSE, smoothness, and bound penalties
- **Data preprocessing**: Frame stacking, normalization, episode splitting
- **Model architectures**: Simple and large MLPs with regularization
- **Evaluation tools**: GUI playback with performance metrics

## Quick Start

### 🚀 YAML-Based Pipeline (Recommended)
```bash
# Use default configuration
python -m passive_walker.bc.run_pipeline

# Use preset configurations
python -m passive_walker.bc.run_pipeline --preset quick_test
python -m passive_walker.bc.run_pipeline --preset full_control
python -m passive_walker.bc.run_pipeline --preset advanced

# Use custom configuration file
python -m passive_walker.bc.run_pipeline --config my_config.yaml
```

### 📋 Step-by-Step Manual Process

#### 1. Collect FSM Demonstration Data
```bash
# Collect FSM walking demonstrations
python -m passive_walker.fsm.collect --episodes 200 --steps 2000 --out data/fsm_runs
```

#### 2. Train BC Model
```bash
# Train hip-only model (PyTorch)
python -m passive_walker.bc.train --backend torch --section hip --data data/fsm_runs --epochs 50

# Train full control model (JAX)
python -m passive_walker.bc.train --backend jax --section both --data data/fsm_runs --epochs 30
```

#### 3. Evaluate Model
```bash
# Play trained model with GUI
python -m passive_walker.bc.play --ckpt checkpoints/torch_hip_seed123.pt --meta checkpoints/torch_hip_seed123_meta.json --episodes 5 --gui

# Headless evaluation
python -m passive_walker.bc.play --ckpt checkpoints/jax_both_seed123.eqx --meta checkpoints/jax_both_seed123_meta.json --episodes 10 --no-gui
```

## File Structure

### Core Files
- **`run_pipeline.py`** — **Complete end-to-end pipeline** (data collection + training + evaluation + plotting)
- **`train.py`** — Unified training CLI with PyTorch and JAX support
- **`play.py`** — Model evaluation and playback with GUI
- **`dataset.py`** — Data loading, preprocessing, and validation
- **`utils.py`** — Utilities for seeding, normalization, checkpointing

### Model Files
- **`models_torch.py`** — PyTorch MLP architectures (simple and large)
- **`models_jax.py`** — JAX/Equinox MLP architectures

### Advanced Training
- **`two_stage_train.py`** — Advanced two-stage training pipeline for improved performance

### Configuration
- **`bc_train.yaml`** — Training configuration presets
- **`checkpoints/`** — Saved models and metadata

## 🚀 Complete Pipeline

The `run_pipeline.py` script provides a complete end-to-end workflow:

### Pipeline Features
- **Automatic data collection** if not available
- **Model training** with configurable parameters
- **Model evaluation** with performance metrics
- **Automatic plotting** of training and evaluation results
- **Summary reports** with all results and configurations

### Pipeline Parameters (Most Important First)

#### 🎯 Control Section (MOST IMPORTANT)
- `--section hip` — Control only hip joint (FSM controls knees)
- `--section knees` — Control only knee joints (FSM controls hip)  
- `--section both` — Control all joints (full BC)
- `--section both-adv` — Full control with advanced loss function

#### 🔧 Backend & Training
- `--backend {torch,jax}` — Training backend
- `--episodes N` — Number of FSM episodes to collect
- `--epochs N` — Training epochs
- `--batch N` — Batch size
- `--lr FLOAT` — Learning rate

#### 📊 Data & Evaluation
- `--steps N` — Steps per episode
- `--eval-episodes N` — Evaluation episodes
- `--eval-seconds FLOAT` — Max seconds per episode
- `--gui` — Enable GUI for evaluation

#### ⚙️ Advanced Options
- `--frame-stack N` — Temporal context frames
- `--w1, --w2, --w3, --w4` — Advanced loss weights
- `--data-dir PATH` — Data directory
- `--results-dir PATH` — Results directory

### Pipeline Examples
```bash
# Quick hip-only training
python -m passive_walker.bc.run_pipeline --section hip --episodes 100 --epochs 20 --gui

# Full control with JAX
python -m passive_walker.bc.run_pipeline --section both --backend jax --episodes 200 --epochs 30

# Advanced training with custom loss
python -m passive_walker.bc.run_pipeline --section both-adv --episodes 200 --epochs 40 --w1 1.0 --w2 0.1 --w3 0.2 --w4 0.05

# Quick test with minimal data
python -m passive_walker.bc.run_pipeline --section hip --episodes 20 --steps 500 --epochs 5 --eval-episodes 3
```

## Training Options

### Control Sections
- **`hip`** — Control only hip joint (FSM controls knees)
- **`knees`** — Control only knee joints (FSM controls hip)
- **`both`** — Control all joints (full BC)
- **`both-adv`** — Full control with advanced loss function

### Backend Selection
- **PyTorch** — Fast single-environment training, easy debugging
- **JAX** — Vectorized operations, JIT compilation, better for research

### Advanced Features
- **Frame stacking** — Temporal context for better control
- **Data normalization** — Stable training across different scales
- **Early stopping** — Prevent overfitting
- **Checkpointing** — Save best models automatically

## Model Architectures

### Simple MLP (TorchMLP)
- 2 hidden layers with GELU activation
- Tanh output for bounded actions
- Lightweight and fast

### Large MLP (TorchMLPLarge)
- 3 hidden layers with batch normalization
- Dropout for regularization
- Better for complex control policies

## Usage Examples

### Basic Training
```bash
# Train hip-only model
python -m passive_walker.bc.train \
    --backend torch \
    --section hip \
    --data data/fsm_runs \
    --epochs 50 \
    --batch 1024 \
    --lr 1e-3

# Train with advanced loss
python -m passive_walker.bc.train \
    --backend torch \
    --section both-adv \
    --data data/fsm_runs \
    --epochs 30 \
    --w1 1.0 --w2 0.0 --w3 0.1 --w4 0.01
```

### Evaluation
```bash
# GUI evaluation
python -m passive_walker.bc.play \
    --ckpt checkpoints/torch_hip_seed123.pt \
    --meta checkpoints/torch_hip_seed123_meta.json \
    --episodes 5 \
    --seconds 25.0 \
    --gui

# Batch evaluation
python -m passive_walker.bc.play \
    --ckpt checkpoints/jax_both_seed123.eqx \
    --meta checkpoints/jax_both_seed123_meta.json \
    --episodes 100 \
    --no-gui
```

## Troubleshooting

### Common Issues
1. **Model falls immediately** — Try training with more data or larger model
2. **Low success rate** — Check data quality and model architecture
3. **Import errors** — Ensure PyTorch or JAX is installed
4. **Memory issues** — Reduce batch size or use smaller model

### Performance Tips
- Use `both` section for full control instead of `hip`-only
- Increase training data with more FSM episodes
- Try advanced loss function with `both-adv` section
- Use larger model architecture for complex policies

## Advanced Usage

### Two-Stage Training
For improved performance, use the two-stage training pipeline:
```bash
python -m passive_walker.bc.two_stage_train \
    --original-data data/fsm_runs \
    --hip-model checkpoints/torch_hip_seed123.pt \
    --hip-meta checkpoints/torch_hip_seed123_meta.json \
    --output-dir results/two_stage
```

This approach can help overcome coordination issues in BC training by gradually building up control complexity.