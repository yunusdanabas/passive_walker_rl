#!/bin/bash
#
# BC Training Script
#
# Usage:
#   ./train_bc.sh [data_dir] [model_name] [epochs]
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"

# Default arguments
DATA_DIR="${1:-experiments/data/fsm_demos}"
MODEL_NAME="${2:-bc_model}"
EPOCHS="${3:-100}"

echo ""
echo "========================================================================"
echo "  BC TRAINING"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Data: $DATA_DIR"
echo "  - Model: $MODEL_NAME"
echo "  - Epochs: $EPOCHS"
echo "  - Output: experiments/models/bc/$MODEL_NAME"
echo ""

# Navigate to project
cd "$PROJECT_DIR"

# Run training
echo "Starting BC training..."
echo ""

mamba run -n main python -m passive_walker.bc.training.train \
    --data "$DATA_DIR" \
    --out "experiments/models/bc/$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --seed 123

echo ""
echo "========================================================================"
echo "  BC TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Model saved to: experiments/models/bc/$MODEL_NAME"
echo ""
echo "Next steps:"
echo "  1. Evaluate model: python tools/evaluate_model.py \\"
echo "                       experiments/models/bc/$MODEL_NAME.pt --type bc"
echo "  2. Play model: python -m passive_walker.bc.evaluation.play \\"
echo "                     experiments/models/bc/$MODEL_NAME.pt --gui"
echo ""

