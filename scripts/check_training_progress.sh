#!/bin/bash
# Quick training progress checker

echo "========================================="
echo "PPO Improved Training Progress Check"
echo "========================================="
echo ""

# Check if process is running
if ps aux | grep "ppo.train.*ppo_improved" | grep -v grep | grep -v "COMMAND" > /dev/null; then
    echo "Status: RUNNING"
    
    # Get process info
    PID=$(ps aux | grep "ppo.train.*ppo_improved" | grep -v grep | grep "python" | awk '{print $2}')
    if [ -n "$PID" ]; then
        echo "PID: $PID"
        
        # Check CPU usage
        CPU=$(ps -p $PID -o %cpu= | xargs)
        echo "CPU: ${CPU}%"
        
        # Check memory usage
        MEM=$(ps -p $PID -o %mem= | xargs)
        echo "Memory: ${MEM}%"
    fi
else
    echo "Status: COMPLETED or NOT RUNNING"
fi

echo ""
echo "Output directory size:"
ls -lh experiments/ppo_improved/ppo_improved/ 2>/dev/null | grep -v total

echo ""
echo "Check for checkpoints:"
find experiments/ppo_improved -name "*.pth" 2>/dev/null | head -5

echo ""
echo "Timestamps:"
stat experiments/ppo_improved/ppo_improved/events.out.tfevents.* 2>/dev/null | grep Modify

echo ""
echo "========================================="
