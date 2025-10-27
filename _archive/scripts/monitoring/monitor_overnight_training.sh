#!/bin/bash
# Overnight Training Monitoring Script

echo "========================================="
echo "OVERNIGHT PPO TRAINING MONITOR"
echo "========================================="
echo ""

# Check running processes
echo "Active training processes:"
ps aux | grep "ppo.train" | grep -v grep | grep python | awk '{print "PID:", $2, "CPU:", $3"%", "CMD:", substr($0, index($0, "python"))}'

echo ""
echo "Training status:"
if ps aux | grep "ppo.train" | grep -v grep | grep python > /dev/null; then
    echo "✅ Training is running"
    RUNNING_COUNT=$(ps aux | grep "ppo.train" | grep -v grep | grep python | wc -l)
    echo "📊 Active processes: $RUNNING_COUNT"
else
    echo "❌ No training processes running"
fi

echo ""
echo "Completed models:"
find experiments/overnight -name "final_model.pth" 2>/dev/null | while read model; do
    if [ -f "$model" ]; then
        echo "✅ $(basename $(dirname $model))"
    fi
done

echo ""
echo "Output directories:"
ls -lh experiments/overnight/ 2>/dev/null | head -10

echo ""
echo "Recent checkpoints:"
find experiments/overnight -name "*.pth" -exec ls -lh {} \; 2>/dev/null | head -5

echo ""
echo "Training logs:"
find experiments/overnight -name "events.out.tfevents.*" -exec ls -lh {} \; 2>/dev/null | head -5

echo ""
echo "========================================="
echo "Next steps:"
echo "1. Monitor with: watch -n 60 ./monitor_overnight_training.sh"
echo "2. Test completed models: ./test_overnight_models.sh"
echo "3. View tensorboard: tensorboard --logdir=experiments/overnight"
echo "========================================="

