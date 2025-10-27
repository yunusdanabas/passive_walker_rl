# Analysis Code

Unified analysis pipeline for passive walker behavioral cloning models.

## 📁 Structure

```
analysis_code/
├── behavioral_analysis.py      # Control patterns & trajectory analysis
├── robustness_testing.py        # Physics variations & disturbance testing
├── run_analysis_pipeline.py     # MAIN: Unified analysis runner
├── run_model_optimization.py    # Model optimization pipeline
└── results/                     # Temporary results (cleaned on new runs)
```

## 🚀 Quick Start

### Run Complete Analysis

```bash
python tools/analysis/run_analysis_pipeline.py \
    --checkpoint experiments/models/torch_both_seed123_ep1_steps180000.pt \
    --meta experiments/models/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10
```

**Output:**
- `experiments/results/analysis_YYYYMMDD_HHMMSS/figures/` - All visualizations
- `experiments/results/analysis_YYYYMMDD_HHMMSS/data/` - Metrics & metadata
- `experiments/results/latest_analysis/` - Symlink to most recent run

### Run Model Optimization

```bash
python tools/analysis/run_model_optimization.py \
    --config passive_walker/bc/pipeline_config.yaml \
    --components all \
    --max-trials 25
```

**Output:**
- `experiments/results/model_optimization/optimization_YYYYMMDD_HHMMSS/` - All optimization results
- `experiments/results/model_optimization/latest_optimization/` - Symlink to most recent run

## 📊 Analysis Components

### 1. Behavioral Analysis (`behavioral_analysis.py`)

**What it does:**
- Compares NN vs FSM control patterns
- Analyzes joint trajectories and phase portraits
- Visualizes action distributions

**Visual Outputs:**
- `control_patterns.png` - 3x3 grid showing:
  - Action trajectories over time
  - Action value distributions
  - Joint angle phase portraits
- `trajectory_comparison.png` - Performance metrics with:
  - Forward progress plots
  - Reward over time
  - Walking speed comparison
  - Performance summary table

**Text Outputs:**
- Minimal JSON with key metrics only

### 2. Robustness Testing (`robustness_testing.py`)

**What it does:**
- Tests under 6 physics conditions:
  - Baseline
  - Low/high friction
  - Steep ramp
  - Heavy/light mass
- Compares NN vs FSM resilience

**Visual Outputs:**
- `robustness_testing.png` - Multi-panel visualization:
  - Reward comparison bars (all conditions)
  - Success rate comparison
  - Distance boxplots per condition
- `robustness_summary_table.png` - Detailed performance table

**Text Outputs:**
- Minimal JSON with averages only

## 🎯 Design Principles

### Visual-First Approach
- **Maximum visualizations**: Comprehensive plots and tables
- **Minimum text output**: Only essential JSON/MD files
- **Self-contained figures**: All information embedded in plots

### Organization
- **Single output folder** per run with timestamp
- **Separated data types**:
  - `figures/` - All PNG files
  - `data/` - JSON metadata only
- **Latest symlink** for easy access

### Naming Convention
- **Descriptive names**: No "Phase 1/2/3" - use actual purpose
- **Timestamp folders**: `analysis_20251020_103045/`
- **Clear file names**: `control_patterns.png`, `robustness_testing.png`

## 📋 Command Reference

### Analysis Pipeline Options

```bash
python tools/analysis/run_analysis_pipeline.py \
    --checkpoint PATH         # Required: Model checkpoint
    --meta PATH              # Required: Model metadata JSON
    --episodes N             # Default: 10
    --output-dir DIR         # Default: experiments/results
    --skip-behavioral        # Skip behavioral analysis
    --skip-robustness        # Skip robustness testing
```

### Optimization Pipeline Options

```bash
python tools/analysis/run_model_optimization.py \
    --config PATH                    # Required: Pipeline config YAML
    --components COMP [COMP ...]     # hyperparams, architecture, advanced, multiobjective, all
    --max-trials N                   # Default: 25
    --method METHOD                  # grid or random (default: random)
    --output-dir DIR                 # Default: experiments/results/model_optimization
```

## 📈 Output Examples

### Analysis Run Output Structure
```
experiments/results/analysis_20251020_103045/
├── figures/
│   ├── control_patterns.png           # 3x3 behavioral analysis grid
│   ├── trajectory_comparison.png      # Performance comparison
│   ├── robustness_testing.png         # Multi-condition testing
│   ├── robustness_summary_table.png   # Detailed results table
│   └── analysis_report.png            # Combined summary
└── data/
    └── analysis_metadata.json         # Run metadata
```

### Optimization Run Output Structure
```
experiments/results/model_optimization/optimization_20251020_120000/
├── hyperparameters/
│   ├── trial_001/ ... trial_025/
│   └── random_search_final.json
├── architecture/
│   ├── arch_trial_001/ ... arch_trial_012/
│   └── architecture_search_final.json
├── figures/
│   └── optimization_summary.png
└── data/
    └── optimization_metadata.json
```

## 🔄 Integration with Main Project

### File Locations
- **Analysis input**: Models from `experiments/models/`
- **Analysis output**: `experiments/results/` (consolidated)
- **Old results**: Archived in `experiments/results/behavior_analysis/`, `experiments/results/robustness_testing/`, `experiments/results/model_optimization/`

### Workflow
1. Train model → `experiments/models/model.pt`
2. Run analysis → `experiments/results/analysis_TIMESTAMP/`
3. Review visualizations in `figures/`
4. Run optimization → `experiments/results/model_optimization/optimization_TIMESTAMP/`
5. Access latest via symlink → `experiments/results/latest_analysis/`

## 🛠️ Development

### Adding New Analysis
1. Create module (e.g., `failure_analysis.py`)
2. Implement `run_failure_analysis(checkpoint, meta, output_dir, **kwargs)` function
3. Return `{'figures': [...], 'metrics': {...}}`
4. Import in `run_analysis_pipeline.py`

### Adding New Optimization
1. Create optimizer in `passive_walker/bc/optimization/`
2. Import in `run_model_optimization.py`
3. Add component choice in argparse
4. Call optimizer in main flow

## 📝 Notes

- All plots are **high-resolution PNG** (150 DPI)
- JSON files contain **only essential metrics**
- Each run creates **isolated timestamped folder**
- Symlinks point to **latest run** for convenience
- Old `codes/` directory has been **removed and consolidated**
