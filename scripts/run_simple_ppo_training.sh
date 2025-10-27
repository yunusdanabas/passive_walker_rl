#!/bin/bash
#
# Simple PPO Training Script
# Focuses on basic MLP training with research rewards - no complexity
#
# Usage:
#   ./run_simple_ppo_training.sh
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"
OUTPUT_DIR="experiments/runs/ppo"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo "  SIMPLE PPO TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Model: MLP [64, 64]"
echo "  - Timesteps: 200,000 (~5-10 min)"
echo "  - Mode: Research rewards (no curriculum)"
echo "  - Environment: Baseline (no randomization)"
echo "  - Evaluation: Every 10k steps (20 evals total)"
echo ""
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to project
cd "$PROJECT_DIR"

# Run simple training
echo "Starting training..."
echo ""

mamba run -n main python -m passive_walker.ppo.train \
    --experiment_name "ppo_simple_baseline" \
    --model_type mlp \
    --hidden_sizes 64 64 \
    --timesteps 200000 \
    --eval_freq 10000 \
    --learning_rate 3e-4 \
    --n_steps 2048 \
    --batch_size 64 \
    --seed 42 \
    --device cpu \
    --out "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/ppo_simple_baseline"
echo ""
echo "Next steps:"
echo "  1. Check training log above for final metrics"
echo "  2. View TensorBoard: tensorboard --logdir=$OUTPUT_DIR"
echo "  3. Evaluate model: python tools/evaluate_model.py \\"
echo "                       $OUTPUT_DIR/ppo_simple_baseline/final_model.pth --type ppo"
echo ""

