#!/bin/bash
# Sequential Overnight Training - One Model at a Time

set -e

echo "========================================="
echo "SEQUENTIAL OVERNIGHT TRAINING"
echo "========================================="
echo "Training one model at a time for ~12 hours"
echo "Each model gets full system resources"
echo ""

# Configuration functions
config_small() {
    echo "--timesteps 500000 --learning_rate 3e-4 --n_epochs 10 --batch_size 64 --n_steps 2048 --eval_freq 25000 --n_eval_episodes 10"
}

config_medium() {
    echo "--timesteps 1000000 --learning_rate 2e-4 --n_epochs 15 --batch_size 128 --n_steps 4096 --eval_freq 50000 --n_eval_episodes 10"
}

config_large() {
    echo "--timesteps 1500000 --learning_rate 1e-4 --n_epochs 20 --batch_size 256 --n_steps 8192 --eval_freq 75000 --n_eval_episodes 10"
}

config_temporal() {
    echo "--timesteps 1000000 --learning_rate 1e-4 --n_epochs 15 --batch_size 128 --n_steps 4096 --eval_freq 50000 --n_eval_episodes 10"
}

config_advanced() {
    echo "--timesteps 750000 --learning_rate 2e-4 --n_epochs 10 --batch_size 64 --n_steps 2048 --eval_freq 37500 --n_eval_episodes 10"
}

# Function to train a single model
train_model() {
    local model_name=$1
    local model_type=$2
    local hidden_sizes=$3
    local seed=$4
    local config_func=$5
    local expected_duration=$6
    
    echo "========================================="
    echo "Starting: $model_name"
    echo "Type: $model_type"
    echo "Hidden: $hidden_sizes"
    echo "Seed: $seed"
    echo "Expected duration: $expected_duration"
    echo "========================================="
    
    local start_time=$(date +%s)
    
    # Build command based on model type
    if [[ "$model_type" == "mlp" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type mlp \
            --hidden_sizes $hidden_sizes \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            $($config_func)
    elif [[ "$model_type" == "lstm" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type lstm \
            --hidden_size $hidden_sizes \
            --num_layers 1 \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            $($config_func)
    elif [[ "$model_type" == "lstm_deep" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type lstm \
            --hidden_size $hidden_sizes \
            --num_layers 2 \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            $($config_func)
    elif [[ "$model_type" == "gru" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type gru \
            --hidden_size $hidden_sizes \
            --num_layers 1 \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            $($config_func)
    elif [[ "$model_type" == "curriculum" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type mlp \
            --hidden_sizes $hidden_sizes \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            --timesteps 750000 \
            --learning_rate 2e-4 \
            --n_epochs 10 \
            --batch_size 64 \
            --n_steps 2048 \
            --eval_freq 37500 \
            --n_eval_episodes 10 \
            --use_curriculum
    elif [[ "$model_type" == "randomization" ]]; then
        python -m passive_walker.ppo.train \
            --experiment_name "$model_name" \
            --model_type mlp \
            --hidden_sizes $hidden_sizes \
            --seed $seed \
            --device cpu \
            --out experiments/sequential \
            --timesteps 750000 \
            --learning_rate 2e-4 \
            --n_epochs 10 \
            --batch_size 64 \
            --n_steps 2048 \
            --eval_freq 37500 \
            --n_eval_episodes 10 \
            --use_domain_randomization \
            --randomization_profile moderate
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    echo "========================================="
    echo "Completed: $model_name"
    echo "Actual duration: ${duration_min} minutes"
    echo "========================================="
    
    # Test the completed model
    echo "Testing completed model..."
    python play_ppo_model.py \
        --model "experiments/sequential/$model_name/final_model.pth" \
        --episodes 5 \
        --headless \
        --deterministic > "test_results_${model_name}.txt" 2>&1
    
    # Extract and display results
    if [ -f "test_results_${model_name}.txt" ]; then
        SUCCESS_RATE=$(grep "Success rate:" "test_results_${model_name}.txt" | awk '{print $3}' | sed 's/%//')
        AVG_LENGTH=$(grep "Average length:" "test_results_${model_name}.txt" | awk '{print $3}')
        AVG_RETURN=$(grep "Average return:" "test_results_${model_name}.txt" | awk '{print $3}')
        
        echo "Test Results:"
        echo "  Success rate: ${SUCCESS_RATE}%"
        echo "  Average length: ${AVG_LENGTH} steps"
        echo "  Average return: ${AVG_RETURN}"
        
        # Check if target achieved
        if [ ! -z "$SUCCESS_RATE" ] && [ ! -z "$AVG_LENGTH" ]; then
            SUCCESS_NUM=$(echo "$SUCCESS_RATE" | sed 's/%//')
            LENGTH_NUM=$(echo "$AVG_LENGTH" | sed 's/ steps//')
            
            if (( $(echo "$SUCCESS_NUM >= 80" | bc -l) )) && (( $(echo "$LENGTH_NUM >= 2000" | bc -l) )); then
                echo "🎉 TARGET ACHIEVED! 20+ second episodes!"
                echo "   Success rate: ${SUCCESS_RATE}%"
                echo "   Episode length: ${AVG_LENGTH}"
                echo "   This model can walk for 20+ seconds!"
                
                # Run comprehensive test
                echo "Running comprehensive test (20 episodes)..."
                python play_ppo_model.py \
                    --model "experiments/sequential/$model_name/final_model.pth" \
                    --episodes 20 \
                    --headless \
                    --deterministic > "comprehensive_test_${model_name}.txt" 2>&1
                
                echo "🎉 SUCCESS! Training complete with target achieved!"
                echo "Best model: $model_name"
                echo "Run with GUI: python play_ppo_model.py --model experiments/sequential/$model_name/final_model.pth --episodes 3"
                exit 0
            fi
        fi
    fi
    
    echo ""
    echo "Next model starting in 30 seconds..."
    sleep 30
}

# Main sequential training
main() {
    echo "Starting sequential training at $(date)"
    echo ""
    
    # Phase 1: Small Models (2 hours)
    train_model "sequential_small_32" "mlp" "32 32" 42 config_small "30 minutes"
    train_model "sequential_small_64" "mlp" "64 64" 43 config_small "30 minutes"
    train_model "sequential_small_128" "mlp" "128 128" 44 config_small "60 minutes"
    
    # Phase 2: Medium Models (3 hours)
    train_model "sequential_medium_256" "mlp" "256 256" 45 config_medium "60 minutes"
    train_model "sequential_medium_512_256" "mlp" "512 256" 46 config_medium "60 minutes"
    train_model "sequential_medium_deep" "mlp" "64 64 64" 47 config_medium "60 minutes"
    
    # Phase 3: Large Models (3 hours)
    train_model "sequential_large_512" "mlp" "512 512" 48 config_large "60 minutes"
    train_model "sequential_large_1024" "mlp" "1024 512" 49 config_large "60 minutes"
    train_model "sequential_large_deep" "mlp" "256 256 256" 50 config_large "60 minutes"
    
    # Phase 4: Temporal Models (3 hours)
    train_model "sequential_lstm_128" "lstm" "128" 51 config_temporal "45 minutes"
    train_model "sequential_lstm_256" "lstm" "256" 52 config_temporal "45 minutes"
    train_model "sequential_lstm_deep" "lstm_deep" "128" 53 config_temporal "45 minutes"
    train_model "sequential_gru_128" "gru" "128" 54 config_temporal "45 minutes"
    
    # Phase 5: Advanced Configurations (1 hour)
    train_model "sequential_curriculum" "curriculum" "256 256" 55 config_advanced "30 minutes"
    train_model "sequential_randomization" "randomization" "256 256" 56 config_advanced "30 minutes"
    
    echo "========================================="
    echo "ALL MODELS COMPLETED!"
    echo "========================================="
    echo "Training finished at $(date)"
    echo "Total duration: ~12 hours"
    echo ""
    echo "Check results:"
    echo "  ls test_results_*.txt"
    echo "  ./test_sequential_models.sh"
    echo ""
    echo "Best models to test with GUI:"
    echo "  python play_ppo_model.py --model experiments/sequential/sequential_lstm_256/final_model.pth --episodes 3"
    echo "  python play_ppo_model.py --model experiments/sequential/sequential_curriculum/final_model.pth --episodes 3"
    echo "========================================="
}

# Run the sequential training
main

