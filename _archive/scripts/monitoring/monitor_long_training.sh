#!/bin/bash
# Monitor long training progress

echo "========================================="
echo "LONG WALKER TRAINING MONITOR"
echo "========================================="
echo ""

# Check processes
echo "Training processes:"
ps aux | grep "ppo.train" | grep -v grep | grep python | awk '{print "PID:", $2, "CPU:", $3"%", "CMD:", substr($0, index($0, "python"))}'

echo ""
echo "Output directories:"
echo "1M timesteps (ppo_long_walker):"
ls -lh experiments/ppo_long/ppo_long_walker/ 2>/dev/null | head -5

echo ""
echo "Exploration (ppo_exploration):"
ls -lh experiments/ppo_exploration/ppo_exploration/ 2>/dev/null | head -5

echo ""
echo "Check for checkpoints:"
find experiments/ppo_long experiments/ppo_exploration -name "*.pth" 2>/dev/null | head -5

echo ""
echo "========================================="
