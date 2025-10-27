#!/bin/bash
# Sequential Model Testing Script

echo "========================================="
echo "SEQUENTIAL MODEL TESTING"
echo "========================================="

# Find all completed models
MODELS=$(find experiments/sequential -name "final_model.pth" 2>/dev/null)

if [ -z "$MODELS" ]; then
    echo "❌ No completed models found"
    echo "Run training first: ./start_sequential_training.sh"
    exit 1
fi

echo "Found $(echo "$MODELS" | wc -l) completed models"
echo ""

# Test each model
for model in $MODELS; do
    EXPERIMENT_NAME=$(basename $(dirname $model))
    echo "Testing model: $EXPERIMENT_NAME"
    echo "Model path: $model"
    
    # Quick test (5 episodes)
    echo "Running quick test (5 episodes)..."
    python play_ppo_model.py \
        --model "$model" \
        --episodes 5 \
        --headless \
        --deterministic > "test_results_${EXPERIMENT_NAME}.txt" 2>&1
    
    # Extract results
    SUCCESS_RATE=$(grep "Success rate:" "test_results_${EXPERIMENT_NAME}.txt" | awk '{print $3}' | sed 's/%//')
    AVG_LENGTH=$(grep "Average length:" "test_results_${EXPERIMENT_NAME}.txt" | awk '{print $3}')
    AVG_RETURN=$(grep "Average return:" "test_results_${EXPERIMENT_NAME}.txt" | awk '{print $3}')
    
    echo "Results:"
    echo "  Success rate: ${SUCCESS_RATE}%"
    echo "  Average length: ${AVG_LENGTH} steps"
    echo "  Average return: ${AVG_RETURN}"
    
    # Check if this model meets our criteria
    if [ ! -z "$SUCCESS_RATE" ] && [ ! -z "$AVG_LENGTH" ]; then
        SUCCESS_NUM=$(echo "$SUCCESS_RATE" | sed 's/%//')
        LENGTH_NUM=$(echo "$AVG_LENGTH" | sed 's/ steps//')
        
        if (( $(echo "$SUCCESS_NUM >= 80" | bc -l) )) && (( $(echo "$LENGTH_NUM >= 2000" | bc -l) )); then
            echo "🎉 EXCELLENT! This model meets our criteria!"
            echo "   - Success rate: ${SUCCESS_RATE}% (target: 80%+)"
            echo "   - Episode length: ${AVG_LENGTH} (target: 2000+ steps)"
            echo "   - This model can walk for 20+ seconds!"
            
            # Run comprehensive test
            echo "Running comprehensive test (20 episodes)..."
            python play_ppo_model.py \
                --model "$model" \
                --episodes 20 \
                --headless \
                --deterministic > "comprehensive_test_${EXPERIMENT_NAME}.txt" 2>&1
            
            COMP_SUCCESS_RATE=$(grep "Success rate:" "comprehensive_test_${EXPERIMENT_NAME}.txt" | awk '{print $3}' | sed 's/%//')
            COMP_AVG_LENGTH=$(grep "Average length:" "comprehensive_test_${EXPERIMENT_NAME}.txt" | awk '{print $3}')
            COMP_AVG_RETURN=$(grep "Average return:" "comprehensive_test_${EXPERIMENT_NAME}.txt" | awk '{print $3}')
            
            echo "Comprehensive results:"
            echo "  Success rate: ${COMP_SUCCESS_RATE}%"
            echo "  Average length: ${COMP_AVG_LENGTH} steps"
            echo "  Average return: ${COMP_AVG_RETURN}"
            
            # Save best model info
            echo "$EXPERIMENT_NAME,$COMP_SUCCESS_RATE,$COMP_AVG_LENGTH,$COMP_AVG_RETURN" >> best_sequential_models.csv
            
            echo ""
            echo "🎉 BEST MODEL FOUND!"
            echo "Run with GUI:"
            echo "python play_ppo_model.py --model $model --episodes 3"
            
        elif (( $(echo "$SUCCESS_NUM >= 50" | bc -l) )) && (( $(echo "$LENGTH_NUM >= 1000" | bc -l) )); then
            echo "✅ Good! This model shows improvement"
            echo "   - Success rate: ${SUCCESS_RATE}% (target: 80%+)"
            echo "   - Episode length: ${AVG_LENGTH} (target: 2000+ steps)"
        else
            echo "❌ This model needs more training"
            echo "   - Success rate: ${SUCCESS_RATE}% (target: 80%+)"
            echo "   - Episode length: ${AVG_LENGTH} (target: 2000+ steps)"
        fi
    else
        echo "❌ Could not parse results"
    fi
    
    echo ""
    echo "----------------------------------------"
done

echo "========================================="
echo "TESTING COMPLETE"
echo "========================================="

# Show summary
if [ -f "best_sequential_models.csv" ]; then
    echo "Best performing models:"
    echo "Experiment,Success Rate,Avg Length,Avg Return"
    cat best_sequential_models.csv
    echo ""
    echo "To test the best model with GUI:"
    BEST_MODEL=$(head -1 best_sequential_models.csv | cut -d',' -f1)
    echo "python play_ppo_model.py --model experiments/sequential/$BEST_MODEL/final_model.pth --episodes 3"
else
    echo "No models met the 20+ second criteria yet."
    echo "Continue training or check individual results."
fi

echo "========================================="

