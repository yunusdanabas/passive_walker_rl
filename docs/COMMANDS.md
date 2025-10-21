# 🚀 Passive Walker RL - Commands Reference

This file contains all the commonly used commands for running training, analysis and optimization pipelines.

## 📋 Table of Contents

1. [Training Pipeline Commands](#training-pipeline-commands)
2. [Model Testing Commands](#model-testing-commands)
3. [Analysis Commands](#analysis-commands)
4. [Optimization Commands](#optimization-commands)
5. [Quick Reference](#quick-reference)
6. [Monitoring Commands](#monitoring-commands)
7. [File Management](#file-management)

---

## 🎯 Training Pipeline Commands

### Quick Test Training (2-3 minutes)
```bash
python -m passive_walker.bc.run_pipeline --preset quick_test
```

### Basic Training (Default Config)
```bash
python -m passive_walker.bc.run_pipeline --config passive_walker/bc/pipeline_config.yaml
```

### Hip Control Training (Recommended)
```bash
python -m passive_walker.bc.run_pipeline \
    --config passive_walker/bc/pipeline_config.yaml \
    --section hip \
    --epochs 30 \
    --batch 1024 \
    --lr 0.001
```

### Both Joints Training
```bash
python -m passive_walker.bc.run_pipeline \
    --config passive_walker/bc/pipeline_config.yaml \
    --section both \
    --epochs 50 \
    --batch 1024 \
    --lr 0.001
```

### Advanced Training (Both Joints with Advanced Loss)
```bash
python -m passive_walker.bc.run_pipeline \
    --config passive_walker/bc/pipeline_config.yaml \
    --section both-adv \
    --epochs 40 \
    --batch 512 \
    --lr 0.0005 \
    --frame-stack 2
```

### Direct Training (Low-level) - CORRECTED
```bash
python -m passive_walker.bc.train \
    --backend torch \
    --section hip \
    --data data/fsm_demos \
    --label-type qdes \
    --epochs 30 \
    --batch 1024 \
    --lr 0.001 \
    --seed 123 \
    --save-dir checkpoints_fixed
```

### Advanced Hip Training (Best Performance)
```bash
python -m passive_walker.bc.run_pipeline --preset advanced_hip
```

### Custom Configuration Training
```bash
python -m passive_walker.bc.run_pipeline \
    --config passive_walker/bc/pipeline_config.yaml \
    --section hip \
    --backend torch \
    --episodes 500 \
    --epochs 100 \
    --batch 512 \
    --lr 0.0005 \
    --frame-stack 3
```

### Retrain Individual Models (WORKING VERSIONS)
```bash
# Hip-only model (working)
python -m passive_walker.bc.train \
    --backend torch --section hip --data data/fsm_demos \
    --label-type qdes --epochs 30 --batch 1024 --lr 0.001 \
    --seed 123 --save-dir checkpoints_fixed

# Knees-only model (working)  
python -m passive_walker.bc.train \
    --backend torch --section knees --data data/fsm_demos \
    --label-type qdes --epochs 30 --batch 1024 --lr 0.001 \
    --seed 123 --save-dir checkpoints_fixed

# Both joints model (working)
python -m passive_walker.bc.train \
    --backend torch --section both --data data/fsm_demos \
    --label-type qdes --epochs 30 --batch 1024 --lr 0.001 \
    --seed 123 --save-dir checkpoints_fixed
```

---

## 🎮 Model Testing Commands

**✅ WORKING MODELS** (in `checkpoints_fixed/`):
- `torch_hip_seed123_ep1_steps180000.pt` - Hip control only
- `torch_knees_seed123_ep1_steps180000.pt` - Knees control only  
- `torch_both_seed123_ep1_steps180000.pt` - Both joints control

All models trained with correct `label_type="qdes"` parameter.

### Test Single Model (Visual)
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3 \
    --seconds 25.0 \
    --gui
```

### Test Model (Headless - Fast)
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 5 \
    --no-gui
```

### Test Multiple Episodes
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10 \
    --seconds 30.0 \
    --gui
```

### Test with Custom Seed
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3 \
    --seed 456 \
    --gui
```

### Test Frame-Stacked Model
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3 \
    --frame-stack 3 \
    --gui
```

### Quick Model Comparison
```bash
# Test Model 1
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 1 --seconds 20.0 --gui

# Test Model 2 (knees control)
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_knees_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_knees_seed123_ep1_steps180000_meta.json \
    --episodes 1 --seconds 20.0 --gui
```

---

## 🔬 Analysis Commands

### Quick Test Analysis (5-6 minutes)
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3
```

### Standard Analysis (15-20 minutes)
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10
```

### Publication-Quality Analysis (30-35 minutes)
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 20
```

### Behavioral Analysis Only (7-10 minutes)
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10 \
    --skip-robustness
```

### Robustness Testing Only (10-15 minutes)
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10 \
    --skip-behavioral
```

---

## 🔧 BC Diagnostic Commands

### Analyze Training Data
```bash
python -m analysis.bc_data_inspector data/fsm_demos --samples 5
```

### Analyze Model Outputs
```bash
python -m analysis.bc_model_diagnostics \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --data data/fsm_demos
```

### Diagnose Model Actions
```bash
python -m analysis.bc_action_diagnostics \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --duration 5.0
```

### Analyze Dataset Quality
```bash
python -m analysis.bc_data_analysis data/fsm_training --output analysis_results --save-json
```

---

## 🎯 Optimization Commands

### Quick Optimization Test (2 hours)
```bash
python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components all \
    --max-trials 5
```

### Hyperparameter Optimization Only (3-5 hours)
```bash
python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components hyperparams \
    --max-trials 15 \
    --method random
```

### Architecture Search Only (3-4 hours)
```bash
python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components architecture \
    --max-trials 12
```

### Full Optimization - Foreground Run (20-35 hours)
```bash
python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components all \
    --max-trials 25 \
    --method random
```

### Full Optimization - Background Overnight Run (20-35 hours)
```bash
nohup python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components all \
    --max-trials 25 \
    --method random \
    > optimization.log 2>&1 &
```

### Multi-Objective Optimization Only (2-3 hours)
```bash
python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components multiobjective
```

---

## ⚡ Quick Reference

### Most Common Commands

**Quick Training Test:**
```bash
python -m passive_walker.bc.run_pipeline --preset quick_test
```

**Standard Training:**
```bash
python -m passive_walker.bc.run_pipeline \
    --config passive_walker/bc/pipeline_config.yaml \
    --section hip --epochs 30
```

**Visual Model Test:**
```bash
python -m passive_walker.bc.play \
    --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3 --gui
```

**Test All Working Models:**
```bash
# Test hip model
python -m passive_walker.bc.play --ckpt checkpoints_fixed/torch_hip_seed123_ep1_steps180000.pt --meta checkpoints_fixed/torch_hip_seed123_ep1_steps180000_meta.json --episodes 1 --gui

# Test knees model  
python -m passive_walker.bc.play --ckpt checkpoints_fixed/torch_knees_seed123_ep1_steps180000.pt --meta checkpoints_fixed/torch_knees_seed123_ep1_steps180000_meta.json --episodes 1 --gui

# Test both joints model
python -m passive_walker.bc.play --ckpt checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json --episodes 1 --gui
```

**Quick Analysis Test:**
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 3
```

**Full Analysis:**
```bash
python -m analysis.run_analysis_pipeline \
    --checkpoint checkpoints_fixed/torch_both_seed123_ep1_steps180000.pt \
    --meta checkpoints_fixed/torch_both_seed123_ep1_steps180000_meta.json \
    --episodes 10
```

**Overnight Optimization:**
```bash
nohup python -m analysis.run_model_optimization \
    --config passive_walker/bc/pipeline_config.yaml \
    --components all \
    --max-trials 25 \
    --method random \
    > optimization.log 2>&1 &
```

---

## 🔍 Monitoring Commands

### Check Running Processes
```bash
# Check if optimization is running
ps aux | grep run_model_optimization

# Check if analysis is running
ps aux | grep run_analysis_pipeline
```

### Monitor Progress (Background Runs)
```bash
# Monitor optimization log
tail -f optimization.log

# Monitor with timestamps
tail -f optimization.log | while read line; do echo "$(date): $line"; done
```

### Check Results
```bash
# Latest analysis results
ls -la results/latest_analysis/

# Latest optimization results
ls -la results/model_optimization/latest_optimization/

# Check results with sizes
ls -lh results/latest_analysis/figures/
ls -lh results/model_optimization/latest_optimization/figures/
```

---

## 📁 File Management

### Clean Previous Results
```bash
# Clean all analysis runs
rm -rf results/analysis_*/

# Clean all optimization runs
rm -rf results/model_optimization/optimization_*/

# Clean logs
rm -f optimization.log

# Clean all previous results (be careful!)
rm -rf results/*
```

### Archive Results
```bash
# Archive current results with timestamp
tar -czf results_backup_$(date +%Y%m%d_%H%M%S).tar.gz results/

# Archive specific run
tar -czf analysis_$(date +%Y%m%d_%H%M%S).tar.gz results/latest_analysis/
```

---

## 📊 Expected Outputs

### Training Results
**Location:** `experiments/models/` or custom `--save-dir`

**Files Generated:**
- `torch_hip_seed123_ep1_steps180000.pt` - Trained model weights
- `torch_hip_seed123_ep1_steps180000_meta.json` - Training metadata
- `torch_hip_seed123_metrics.json` - Training metrics (loss, accuracy)
- Console output with training progress and final performance

**Training Stages:**
1. **Data Collection** - FSM episodes for training data
2. **Model Training** - Neural network training with BC loss
3. **Model Evaluation** - Performance testing on environment

### Analysis Results
**Location:** `results/latest_analysis/`

**Files Generated:**
- `figures/control_patterns.png` - Action comparisons, reward breakdown
- `figures/trajectory_comparison.png` - Joint angles, velocities over time
- `figures/robustness_testing.png` - Performance across conditions
- `figures/robustness_summary_table.png` - Numerical comparison table
- `figures/analysis_report.png` - Combined summary visualization
- `data/analysis_metadata.json` - All metrics in JSON format

### Optimization Results
**Location:** `results/model_optimization/latest_optimization/`

**Files Generated:**
- `hyperparameters/` - All hyperparameter trial results
- `architecture/` - Architecture search results
- `advanced_training/` - Curriculum learning results
- `multiobjective/` - Multi-objective optimization results
- `figures/optimization_summary.png` - Summary visualization
- `data/optimization_metadata.json` - Complete optimization metadata

---

## 🛠️ Troubleshooting

### Environment Setup
```bash
# Activate environment
mamba activate main

# Check Python path
python -c "import sys; print(sys.path)"

# Test imports
python -c "from passive_walker.core.env import PassiveWalkerEnv; print('✅ Environment OK')"
```

### Common Issues

**Missing Checkpoint Files:**
```bash
# Check available checkpoints
ls -la experiments/models/

# Use available checkpoint
python -m tools.analysis.run_analysis_pipeline \
    --checkpoint experiments/models/YOUR_CHECKPOINT_HERE.pt \
    --meta experiments/models/YOUR_META_HERE.json \
    --episodes 3
```

**Permission Issues:**
```bash
# Fix permissions for results directory
chmod -R 755 results/
```

**Background Process Issues:**
```bash
# Kill background optimization if needed
pkill -f run_model_optimization

# Check what's using ports/processes
netstat -tulpn | grep python
```

---

## 📝 Notes

- **Training Pipeline:** Trains BC models using FSM demonstration data with configurable hyperparameters
- **Analysis Pipeline:** Compares NN vs FSM performance with comprehensive visualizations  
- **Optimization Pipeline:** Searches for best hyperparameters, architectures, and training strategies
- **Runtime Estimates:** Based on standard hardware; actual times may vary
- **Output Organization:** Results are timestamped and organized in `experiments/results/` and `experiments/models/` directories
- **Latest Symlinks:** `results/latest_analysis/` and `results/model_optimization/latest_optimization/` always point to most recent runs

### Training Tips
- **Hip control** (`--section hip`) is typically most stable for initial training
- **Both joints** (`--section both`) is more complex but can achieve better performance
- Use `--preset advanced_hip` for best performance training configuration
- Monitor training loss - should decrease consistently for good training

---

*Last updated: $(date)*
