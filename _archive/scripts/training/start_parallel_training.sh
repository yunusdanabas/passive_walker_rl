#!/bin/bash
# Parallel Overnight Training - All Models Simultaneously

set -e

echo "========================================="
echo "PARALLEL OVERNIGHT TRAINING"
echo "========================================="
echo "Starting ALL 17 models in parallel..."
echo "Expected completion: 4-6 hours (vs 16 hours sequential)"
echo ""

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

# Start all models in parallel
start_all_parallel() {
    echo "Starting Phase 1: Small Models"
    
    # Small MLP [32, 32]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_small_32 \
        --model_type mlp \
        --hidden_sizes 32 32 \
        --timesteps 1000000 \
        --seed 42 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_a) &
    
    # Small MLP [64, 64]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_small_64 \
        --model_type mlp \
        --hidden_sizes 64 64 \
        --timesteps 1000000 \
        --seed 43 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_a) &
    
    # Small MLP [128, 128]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_small_128 \
        --model_type mlp \
        --hidden_sizes 128 128 \
        --timesteps 1000000 \
        --seed 44 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_a) &
    
    echo "Starting Phase 2: Medium Models"
    
    # Medium MLP [256, 256]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_medium_256 \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2000000 \
        --seed 45 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_b) &
    
    # Medium MLP [512, 256]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_medium_512_256 \
        --model_type mlp \
        --hidden_sizes 512 256 \
        --timesteps 2000000 \
        --seed 46 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_b) &
    
    # Medium MLP [64, 64, 64] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name parallel_medium_deep \
        --model_type mlp \
        --hidden_sizes 64 64 64 \
        --timesteps 2000000 \
        --seed 47 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_b) &
    
    echo "Starting Phase 3: Large Models"
    
    # Large MLP [512, 512]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_large_512 \
        --model_type mlp \
        --hidden_sizes 512 512 \
        --timesteps 2000000 \
        --seed 48 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # Large MLP [1024, 512]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_large_1024 \
        --model_type mlp \
        --hidden_sizes 1024 512 \
        --timesteps 2000000 \
        --seed 49 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # Large MLP [256, 256, 256] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name parallel_large_deep \
        --model_type mlp \
        --hidden_sizes 256 256 256 \
        --timesteps 2000000 \
        --seed 50 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    echo "Starting Phase 4: Temporal Models"
    
    # LSTM [128]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_lstm_128 \
        --model_type lstm \
        --hidden_size 128 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 51 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # LSTM [256]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_lstm_256 \
        --model_type lstm \
        --hidden_size 256 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 52 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # LSTM [128, 128] - Deep
    python -m passive_walker.ppo.train \
        --experiment_name parallel_lstm_deep \
        --model_type lstm \
        --hidden_size 128 \
        --num_layers 2 \
        --timesteps 1500000 \
        --seed 53 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # GRU [128]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_gru_128 \
        --model_type gru \
        --hidden_size 128 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 54 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    # GRU [256]
    python -m passive_walker.ppo.train \
        --experiment_name parallel_gru_256 \
        --model_type gru \
        --hidden_size 256 \
        --num_layers 1 \
        --timesteps 1500000 \
        --seed 55 \
        --device cpu \
        --out experiments/overnight_parallel \
        $(config_c) &
    
    echo "Starting Phase 5: Advanced Configurations"
    
    # Curriculum learning
    python -m passive_walker.ppo.train \
        --experiment_name parallel_curriculum \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2500000 \
        --seed 56 \
        --device cpu \
        --out experiments/overnight_parallel \
        --learning_rate 3e-4 \
        --n_epochs 10 \
        --batch_size 64 \
        --n_steps 2048 \
        --eval_freq 50000 \
        --n_eval_episodes 10 \
        --use_curriculum &
    
    # Domain randomization
    python -m passive_walker.ppo.train \
        --experiment_name parallel_randomization \
        --model_type mlp \
        --hidden_sizes 256 256 \
        --timesteps 2000000 \
        --seed 57 \
        --device cpu \
        --out experiments/overnight_parallel \
        --learning_rate 2e-4 \
        --n_epochs 15 \
        --batch_size 128 \
        --n_steps 4096 \
        --eval_freq 100000 \
        --n_eval_episodes 10 \
        --use_domain_randomization \
        --randomization_profile moderate &
    
    # Combined approach
    python -m passive_walker.ppo.train \
        --experiment_name parallel_combined \
        --model_type lstm \
        --hidden_size 256 \
        --num_layers 2 \
        --timesteps 2500000 \
        --seed 58 \
        --device cpu \
        --out experiments/overnight_parallel \
        --learning_rate 3e-4 \
        --n_epochs 10 \
        --batch_size 64 \
        --n_steps 2048 \
        --eval_freq 50000 \
        --n_eval_episodes 10 \
        --use_curriculum \
        --use_domain_randomization \
        --randomization_profile moderate &
    
    echo ""
    echo "========================================="
    echo "ALL 17 MODELS STARTED IN PARALLEL!"
    echo "========================================="
    echo "Expected completion: 4-6 hours"
    echo "Monitor with: ./monitor_parallel_training.sh"
    echo "Test models: ./test_parallel_models.sh"
    echo "========================================="
}

# Run the parallel training
start_all_parallel

