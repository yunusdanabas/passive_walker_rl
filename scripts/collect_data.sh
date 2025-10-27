#!/bin/bash
#
# FSM Data Collection Script
#
# Usage:
#   ./collect_data.sh [episodes] [duration] [output_dir]
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"

# Default arguments
EPISODES="${1:-10}"
DURATION="${2:-20}"
OUTPUT_DIR="${3:-experiments/data/fsm_demos}"

echo ""
echo "========================================================================"
echo "  FSM DATA COLLECTION"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Episodes: $EPISODES"
echo "  - Duration: $DURATION seconds per episode"
echo "  - Output: $OUTPUT_DIR"
echo ""

# Navigate to project
cd "$PROJECT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run collection
echo "Starting data collection..."
echo ""

mamba run -n main python -m passive_walker.fsm.collect \
    --episodes "$EPISODES" \
    --duration "$DURATION" \
    --out "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  DATA COLLECTION COMPLETE"
echo "========================================================================"
echo ""
echo "Data saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Train BC model: ./scripts/train_bc.sh $OUTPUT_DIR"
echo "  2. Inspect data: ls -lh $OUTPUT_DIR"
echo ""

