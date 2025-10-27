#!/bin/bash
# Monitor PPO training progress

echo "Checking training progress..."
ps aux | grep "python.*ppo.train.*ppo_improved" | grep -v grep

echo ""
echo "Recent log entries:"
tail -50 /tmp/ppo_training.log 2>/dev/null || echo "No log file found"

echo ""
echo "Output directory contents:"
ls -lth experiments/ppo_improved/ppo_improved/ 2>/dev/null | head -10
