#!/bin/bash
# Sequential Training Monitor

echo "========================================="
echo "SEQUENTIAL TRAINING MONITOR - $(date)"
echo "========================================="

# Check if training is running
if ps aux | grep "ppo.train" | grep -v grep > /dev/null; then
    echo "✅ Training is active"
    CURRENT_MODEL=$(ps aux | grep "ppo.train" | grep -v grep | grep python | awk '{print $NF}' | head -1)
    echo "📊 Current model: $CURRENT_MODEL"
    
    # Show process details
    ps aux | grep "ppo.train" | grep -v grep | grep python | awk '{print "PID:", $2, "CPU:", $3"%", "Memory:", $4"%", "Time:", $10}'
else
    echo "❌ No training processes running"
fi

echo ""
echo "Completed models:"
COMPLETED_COUNT=$(find experiments/sequential -name "final_model.pth" 2>/dev/null | wc -l)
echo "✅ Completed models: $COMPLETED_COUNT"

if [ $COMPLETED_COUNT -gt 0 ]; then
    find experiments/sequential -name "final_model.pth" 2>/dev/null | while read model; do
        EXPERIMENT_NAME=$(basename $(dirname $model))
        echo "  ✅ $EXPERIMENT_NAME"
    done
fi

echo ""
echo "Test results:"
if [ -f "test_results_*.txt" ]; then
    ls -lh test_results_*.txt 2>/dev/null | head -5
else
    echo "No test results yet"
fi

echo ""
echo "Training progress:"
ls -lh experiments/sequential/ 2>/dev/null | head -10

echo ""
echo "Recent checkpoints:"
find experiments/sequential -name "*.pth" -exec ls -lh {} \; 2>/dev/null | head -5

echo ""
echo "========================================="
echo "Commands:"
echo "  Monitor: ./monitor_sequential_training.sh"
echo "  Test: ./test_sequential_models.sh"
echo "  Logs: tail -f sequential_training.log"
echo "========================================="

