#!/bin/bash
# Overnight PPO Training Scripts

set -e

# Configuration functions
config_a() {
    echo "--learning_rate 3e-4 --n_epochs 10 --batch_size 64 --n_steps 2048 --eval_freq 50000 --n_eval_episodes 10"
}

config_b() {
    echo "--learning_rate 2e-4 --n_epochs 15 --batch_size 128 --n_steps 4096 --eval_freq 100000 --n_eval_episodes 10"
}

config_c() {
    echo "--learning_rate 1e-4 --n_epochs 20 --batch_size 256 --n_steps 8192 --eval_freq 75000 --n_eval_episodes 10"
}

config_d() {
    echo "--learning_rate 3e-4 --n_epochs 10 --batch_size 64 --n_steps 2048 --eval_freq 50000 --n_eval_episodes 10 --use_curriculum"
}

config_e() {
    echo "--learning_rate 2e-4 --n_epochs 15 --batch_size 128 --n_steps 4096 --eval_freq 100000 --n_eval_episodes 10 --use_domain_randomization --randomization_profile moderate"
}

# Phase 1: Small Models
start_phase1() {
    echo "Starting Phase 1: Small Models (3 hours)"
    
    # Small MLP [32, 32]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_small_32 \
        --model_type mlp \
        --hidden_sizes 32 32 \
        --timesteps 1000000 \
        --seed 42 \
        --device cpu \
        --out experiments/overnight \
        $(config_a) &
    
    sleep 30
    
    # Small MLP [64, 64]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_small_64 \
        --model_type mlp \
        --hidden_sizes 64 64 \
        --timesteps 1000000 \
        --seed 43 \
        --device cpu \
        --out experiments/overnight \
        $(config_a) &
    
    sleep 30
    
    # Small MLP [128, 128]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_small_128 \
        --model_type mlp \
        --hidden_sizes 128 128 \
        --timesteps 1000000 \
        --seed 44 \
        --device cpu \
        --out experiments/overnight \
        $(config_a) &
    
    echo "Phase 1 started. Expected completion: 3 hours"
}

# Phase 2: Medium Models
start_phase2() {
    echo "Starting Phase 2: Medium Models (3 hours)"
    
    # Medium MLP [256, 256]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_medium_256 \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2000000 \
        --seed 45 \
        --device cpu \
        --out experiments/overnight \
        $(config_b) &
    
    sleep 30
    
    # Medium MLP [512, 256]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_medium_512_256 \
        --model_type mlp \
        --hidden_sizes 512 256 \
        --timesteps 2000000 \
        --seed 46 \
        --device cpu \
        --out experiments/overnight \
        $(config_b) &
    
    sleep 30
    
    # Medium MLP [64, 64, 64] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name overnight_medium_deep \
        --model_type mlp \
        --hidden_sizes 64 64 64 \
        --timesteps 2000000 \
        --seed 47 \
        --device cpu \
        --out experiments/overnight \
        $(config_b) &
    
    echo "Phase 2 started. Expected completion: 3 hours"
}

# Phase 3: Large Models
start_phase3() {
    echo "Starting Phase 3: Large Models (3 hours)"
    
    # Large MLP [512, 512]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_large_512 \
        --model_type mlp \
        --hidden_sizes 512 512 \
        --timesteps 2000000 \
        --seed 48 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # Large MLP [1024, 512]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_large_1024 \
        --model_type mlp \
        --hidden_sizes 1024 512 \
        --timesteps 2000000 \
        --seed 49 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # Large MLP [256, 256, 256] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name overnight_large_deep \
        --model_type mlp \
        --hidden_sizes 256 256 256 \
        --timesteps 2000000 \
        --seed 50 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    echo "Phase 3 started. Expected completion: 3 hours"
}

# Phase 4: Temporal Models
start_phase4() {
    echo "Starting Phase 4: Temporal Models (3 hours)"
    
    # LSTM [128]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_lstm_128 \
        --model_type lstm \
        --hidden_size 128 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 51 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # LSTM [256]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_lstm_256 \
        --model_type lstm \
        --hidden_size 256 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 52 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # LSTM [128, 128] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name overnight_lstm_deep \
        --model_type lstm \
        --hidden_size 128 \
        --num_layers 2 \
        --timesteps 1500000 \
        --seed 53 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # GRU [128]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_gru_128 \
        --model_type gru \
        --hidden_size 128 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 54 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    sleep 30
    
    # GRU [256]
    python -m passive_walker.ppo.train \
        --experiment_name overnight_gru_256 \
        --model_type gru \
        --hidden_size 256 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 55 \
        --device cpu \
        --out experiments/overnight \
        $(config_c) &
    
    echo "Phase 4 started. Expected completion: 3 hours"
}

# Phase 5: Advanced Configurations
start_phase5() {
    echo "Starting Phase 5: Advanced Configurations (4 hours)"
    
    # Curriculum learning
    python -m passive_walker.ppo.train \
        --experiment_name overnight_curriculum \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2500000 \
        --seed 56 \
        --device cpu \
        --out experiments/overnight \
        $(config_d) &
    
    sleep 30
    
    # Domain randomization
    python -m passive_walker.ppo.train \
        --experiment_name overnight_randomization \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2000000 \
        --seed 57 \
        --device cpu \
        --out experiments/overnight \
        $(config_e) &
    
    sleep 30
    
    # Combined approach
    python -m passive_walker.ppo.train \
        --experiment_name overnight_combined \
        --model_type lstm \
        --hidden_size 256 \
        --num_layers 2 \
        --timesteps 2500000 \
        --seed 58 \
        --device cpu \
        --out experiments/overnight \
        --learning_rate 3e-4 \
        --n_epochs 10 \
        --batch_size 64 \
        --n_steps 2048 \
        --eval_freq 50000 \
        --n_eval_episodes 10 \
        --use_curriculum \
        --use_domain_randomization \
        --randomization_profile moderate &
    
    echo "Phase 5 started. Expected completion: 4 hours"
}

# Main function
main() {
    case "$1" in
        "phase1")
            start_phase1
            ;;
        "phase2")
            start_phase2
            ;;
        "phase3")
            start_phase3
            ;;
        "phase4")
            start_phase4
            ;;
        "phase5")
            start_phase5
            ;;
        "all")
            echo "Starting all phases sequentially..."
            start_phase1
            sleep 10800  # Wait 3 hours
            start_phase2
            sleep 10800  # Wait 3 hours
            start_phase3
            sleep 10800  # Wait 3 hours
            start_phase4
            sleep 10800  # Wait 3 hours
            start_phase5
            ;;
        *)
            echo "Usage: $0 {phase1|phase2|phase3|phase4|phase5|all}"
            echo ""
            echo "Phases:"
            echo "  phase1 - Small models (3 hours)"
            echo "  phase2 - Medium models (3 hours)"
            echo "  phase3 - Large models (3 hours)"
            echo "  phase4 - Temporal models (3 hours)"
            echo "  phase5 - Advanced configurations (4 hours)"
            echo "  all    - All phases sequentially (16 hours)"
            exit 1
            ;;
    esac
}

# Run main function with arguments
main "$@"

