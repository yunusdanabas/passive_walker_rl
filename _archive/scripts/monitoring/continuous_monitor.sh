#!/bin/bash
# Continuous Overnight Training Monitor

echo "========================================="
echo "OVERNIGHT TRAINING STATUS - $(date)"
echo "========================================="

# Check main orchestrator
echo "Main orchestrator:"
ps aux | grep "start_overnight_training" | grep -v grep | awk '{print "PID:", $2, "Status:", $8, "CPU:", $3"%"}'

echo ""
echo "Active PPO training processes:"
ACTIVE_COUNT=$(ps aux | grep "ppo.train" | grep -v grep | grep python | wc -l)
echo "📊 Total active processes: $ACTIVE_COUNT"

if [ $ACTIVE_COUNT -gt 0 ]; then
    ps aux | grep "ppo.train" | grep -v grep | grep python | awk '{print "PID:", $2, "CPU:", $3"%", "Experiment:", $NF}' | head -10
else
    echo "❌ No PPO training processes running"
fi

echo ""
echo "Completed models:"
COMPLETED_COUNT=$(find experiments/overnight -name "final_model.pth" 2>/dev/null | wc -l)
echo "✅ Completed models: $COMPLETED_COUNT"

if [ $COMPLETED_COUNT -gt 0 ]; then
    find experiments/overnight -name "final_model.pth" 2>/dev/null | while read model; do
        EXPERIMENT_NAME=$(basename $(dirname $model))
        echo "  ✅ $EXPERIMENT_NAME"
    done
fi

echo ""
echo "Training progress:"
ls -lh experiments/overnight/ 2>/dev/null | head -10

echo ""
echo "Recent checkpoints:"
find experiments/overnight -name "*.pth" -exec ls -lh {} \; 2>/dev/null | head -5

echo ""
echo "Training logs:"
if [ -f "overnight_training.log" ]; then
    echo "Main log size: $(ls -lh overnight_training.log | awk '{print $5}')"
    echo "Last 5 lines:"
    tail -5 overnight_training.log
else
    echo "No main log file found"
fi

echo ""
echo "========================================="
echo "Next check in 5 minutes..."
echo "========================================="

