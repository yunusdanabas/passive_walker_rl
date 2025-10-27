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

All writers are updated to save into this structure. Legacy paths (e.g., `outputs/`, `results/`, `ppo_plots/`) are auto-redirected with a deprecation warning.
