# Evaluation Scripts

This directory contains scripts for evaluating and comparing BC models.

## Scripts Overview

### `comprehensive_evaluation.py`
**Main evaluation script** - Comprehensive evaluation of all trained models across multiple physics conditions.

**Usage:**
```bash
cd /home/yunusdanabas/passive_walker_rl
python tools/evaluation/comprehensive_evaluation.py
```

**Features:**
- Evaluates all models (Baseline, Enhanced, Gentle, Low_Friction, Mass_Jitter)
- Tests across 6 physics conditions (nominal, gentle, steep, low_friction, high_friction, gentle_low)
- Generates comparison plots and detailed reports
- Creates trajectory visualizations

### `comprehensive_comparison.py`
**Detailed comparison** - In-depth comparison between baseline and enhanced models.

**Usage:**
```bash
cd /home/yunusdanabas/passive_walker_rl
python tools/evaluation/comprehensive_comparison.py
```

**Features:**
- Robustness testing across physics conditions
- Control frequency evaluation (100Hz, 150Hz, 200Hz)
- Detailed performance metrics
- Comprehensive comparison tables

### `evaluate_proper.py`
**Model loading evaluation** - Proper evaluation using actual trained model predictions.

**Usage:**
```bash
cd /home/yunusdanabas/passive_walker_rl
python tools/evaluation/evaluate_proper.py
```

**Features:**
- Loads actual trained models (not random actions)
- Tests both FSM and research modes
- Enhanced reward component analysis
- Proper model inference

### `evaluate_comparison.py`
**Simple comparison** - Basic comparison using existing play.py functionality.

**Usage:**
```bash
cd /home/yunusdanabas/passive_walker_rl
python tools/evaluation/evaluate_comparison.py
```

**Features:**
- Uses existing play.py infrastructure
- Tests with enhanced rewards
- Simple success rate comparison

## Model Paths

All scripts expect models to be located in:
- `experiments/models/baseline/` - Baseline model
- `experiments/models/enhanced/` - Enhanced model
- `experiments/models/gentle/` - Gentle condition model
- `experiments/models/low_friction/` - Low friction model
- `experiments/models/mass_jitter/` - Mass jitter model

## Output

Scripts generate:
- **Plots**: Saved to `experiments/outputs/plots/`
- **Reports**: Saved to `experiments/outputs/reports/`
- **Console output**: Detailed metrics and comparisons

## Recommended Usage

1. **Start with**: `comprehensive_evaluation.py` for full analysis
2. **For detailed comparison**: `comprehensive_comparison.py`
3. **For debugging**: `evaluate_proper.py` to verify model loading
4. **For quick check**: `evaluate_comparison.py` for basic metrics

## Requirements

- All trained models must be present in `experiments/models/` directory
- Environment must be properly configured
- Required Python packages: torch, numpy, matplotlib
