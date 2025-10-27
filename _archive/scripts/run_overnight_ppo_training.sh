#!/bin/bash
#
# Overnight PPO Training Script
# Runs 3 PPO configurations sequentially with full logging
#
# Usage:
#   ./run_overnight_ppo_training.sh        # Run in foreground
#   nohup ./run_overnight_ppo_training.sh > training.log 2>&1 &  # Run in background
#

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"
OUTPUT_DIR="experiments/ppo_runs"
LOG_DIR="experiments/ppo_runs/overnight_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to print section header
print_header() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

# Function to run training with logging
run_training() {
    local exp_name=$1
    local log_file="$LOG_DIR/${exp_name}_${TIMESTAMP}.log"
    
    log "Starting: $exp_name"
    log "Log file: $log_file"
    
    shift  # Remove first argument (exp_name)
    
    # Run training with mamba run (no activation needed)
    if mamba run -n main python -m passive_walker.ppo.train "$@" 2>&1 | tee "$log_file"; then
        log "✅ Completed: $exp_name"
        return 0
    else
        log "❌ Failed: $exp_name"
        return 1
    fi
}

# Start timing
START_TIME=$(date +%s)

print_header "OVERNIGHT PPO TRAINING - STARTED AT $(date)"

# Navigate to project directory
cd "$PROJECT_DIR"
log "Working directory: $(pwd)"

# Display system info (using mamba run)
log "System Information:"
log "  Python: $(mamba run -n main python --version 2>&1)"
log "  PyTorch: $(mamba run -n main python -c 'import torch; print(torch.__version__)' 2>&1)"
log "  CUDA Available: $(mamba run -n main python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
log "  Device: $(mamba run -n main python -c 'import torch; print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))' 2>&1)"

# ============================================================================
# RUN A: MLP Baseline (500k timesteps, ~1-2 hours)
# ============================================================================
print_header "RUN A: MLP BASELINE"
log "Configuration: MLP [64,64], 500k timesteps"
log "Expected duration: 1-2 hours"

RUN_A_STATUS=0
run_training "ppo_mlp_baseline" \
    --experiment_name "ppo_mlp_baseline" \
    --model_type mlp \
    --timesteps 500000 \
    --seed 42 \
    --device cpu \
    --out "$OUTPUT_DIR" || RUN_A_STATUS=$?

if [ $RUN_A_STATUS -ne 0 ]; then
    log "⚠️  Run A failed, but continuing with other runs..."
fi

# ============================================================================
# RUN B: LSTM + Curriculum (1M timesteps, ~3-4 hours)
# ============================================================================
print_header "RUN B: LSTM + CURRICULUM"
log "Configuration: LSTM 128x2, Curriculum, 1M timesteps"
log "Expected duration: 3-4 hours"

RUN_B_STATUS=0
run_training "ppo_lstm_curriculum" \
    --experiment_name "ppo_lstm_curriculum" \
    --model_type lstm \
    --hidden_size 128 \
    --num_layers 2 \
    --use_curriculum \
    --timesteps 1000000 \
    --seed 42 \
    --device cpu \
    --out "$OUTPUT_DIR" || RUN_B_STATUS=$?

if [ $RUN_B_STATUS -ne 0 ]; then
    log "⚠️  Run B failed, but continuing with Run C..."
fi

# ============================================================================
# RUN C: LSTM + Full Enhancement (1M timesteps, ~4-5 hours)
# ============================================================================
print_header "RUN C: LSTM + FULL ENHANCEMENT"
log "Configuration: LSTM 128x2, Curriculum + Aggressive Randomization, 1M timesteps"
log "Expected duration: 4-5 hours"

RUN_C_STATUS=0
run_training "ppo_lstm_advanced" \
    --experiment_name "ppo_lstm_advanced" \
    --model_type lstm \
    --hidden_size 128 \
    --num_layers 2 \
    --use_curriculum \
    --use_domain_randomization \
    --randomization_profile aggressive \
    --timesteps 1000000 \
    --seed 42 \
    --device cpu \
    --out "$OUTPUT_DIR" || RUN_C_STATUS=$?

# ============================================================================
# SUMMARY
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

print_header "OVERNIGHT PPO TRAINING - COMPLETED AT $(date)"

log "Total Duration: ${HOURS}h ${MINUTES}m"
log ""
log "Results Summary:"
log "  Run A (MLP Baseline):           $([ $RUN_A_STATUS -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
log "  Run B (LSTM + Curriculum):      $([ $RUN_B_STATUS -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
log "  Run C (LSTM + Full Enhancement):$([ $RUN_C_STATUS -eq 0 ] && echo '✅ SUCCESS' || echo '❌ FAILED')"
log ""
log "Output directory: $OUTPUT_DIR"
log "Log directory: $LOG_DIR"
log ""

# Generate summary file
SUMMARY_FILE="$LOG_DIR/overnight_summary_${TIMESTAMP}.txt"
cat > "$SUMMARY_FILE" << EOF
====================================================================
OVERNIGHT PPO TRAINING SUMMARY
====================================================================

Start Time: $(date -d @$START_TIME)
End Time:   $(date -d @$END_TIME)
Duration:   ${HOURS}h ${MINUTES}m

--------------------------------------------------------------------
Run Results:
--------------------------------------------------------------------
Run A - MLP Baseline:           $([ $RUN_A_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')
Run B - LSTM + Curriculum:      $([ $RUN_B_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')
Run C - LSTM + Full Enhancement:$([ $RUN_C_STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')

--------------------------------------------------------------------
Output Locations:
--------------------------------------------------------------------
Models:      $OUTPUT_DIR/ppo_mlp_baseline/final_model.pth
             $OUTPUT_DIR/ppo_lstm_curriculum/final_model.pth
             $OUTPUT_DIR/ppo_lstm_advanced/final_model.pth

Logs:        $LOG_DIR/ppo_mlp_baseline_${TIMESTAMP}.log
             $LOG_DIR/ppo_lstm_curriculum_${TIMESTAMP}.log
             $LOG_DIR/ppo_lstm_advanced_${TIMESTAMP}.log

TensorBoard: tensorboard --logdir=$OUTPUT_DIR

--------------------------------------------------------------------
Next Steps:
--------------------------------------------------------------------
1. Check individual run logs for details
2. View TensorBoard for training curves
3. Run evaluation: python -m passive_walker.ppo.evaluate_cli
4. Compare with BC baseline

====================================================================
EOF

log "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

# Exit with appropriate status
if [ $RUN_A_STATUS -ne 0 ] || [ $RUN_B_STATUS -ne 0 ] || [ $RUN_C_STATUS -ne 0 ]; then
    log "⚠️  Some runs failed. Check logs for details."
    exit 1
else
    log "✅ All runs completed successfully!"
    exit 0
fi

