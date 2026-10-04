# Experiments Directory

This directory contains all experimental work for the passive walker RL project.

## Structure (Unified)

- `data/` - Training datasets and collected demonstrations
- `models/` - Trained model checkpoints
  - `bc/`
  - `ppo/`
- `runs/` - Training logs (TensorBoard, etc.)
  - `bc/`
  - `ppo/`
- `analysis/` - All analysis artifacts
  - `metrics/` - JSON metrics and evaluation outputs
  - `plots/` - Visualizations
    - `bc/`
    - `ppo/`
  - `reports/` - Markdown/HTML reports

## Usage

Writers save into this structure. Everything here except this README is git-ignored:
regenerate data and models with the CLIs. Set `PASSIVE_WALKER_HOME` to place `experiments/` elsewhere.
