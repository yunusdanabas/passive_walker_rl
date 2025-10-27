#!/bin/bash
#
# Overnight Training Pipeline
#
# Runs complete pipeline: FSM data collection -> BC training -> PPO training -> Evaluation
#

set -e

# Configuration
PROJECT_DIR="/home/yunusdanabas/passive_walker_rl"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="experiments/overnight_run_${TIMESTAMP}"

echo ""
echo "========================================================================"
echo "  OVERNIGHT TRAINING PIPELINE"
echo "========================================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# =============================================================================
# 1. COLLECT FSM DATA
# =============================================================================
echo ""
echo "========================================================================"
echo "  STEP 1: COLLECTING FSM DATA"
echo "========================================================================"
echo ""

FSM_DATA_DIR="$OUTPUT_DIR/fsm_data"
mkdir -p "$FSM_DATA_DIR"

# Collect FSM data with multiple episodes
mamba run -n main python -m passive_walker.fsm.collect \
    --episodes 100 \
    --duration 25.0 \
    --out "$FSM_DATA_DIR"

echo ""
echo "  FSM data collected to: $FSM_DATA_DIR"
echo ""

# Visualize FSM data
echo "Generating FSM visualizations..."
mamba run -n main python tools/visualize_fsm_data.py "$FSM_DATA_DIR" --output "$OUTPUT_DIR/fsm_data_visualization.png"

# =============================================================================
# 2. TRAIN BC MODELS (SWEEP)
# =============================================================================
echo ""
echo "========================================================================"
echo "  STEP 2: TRAINING BC MODELS (SWEEP)"
echo "========================================================================"
echo ""

BC_DIR="$OUTPUT_DIR/bc_models"
mkdir -p "$BC_DIR"

# BC Sweep Configuration
SEEDS=(123 456 789)
SECTIONS=("hip" "knees" "both")
EPOCHS=50

echo "Training BC models with different seeds and sections..."
echo "  Sections: hip, knees, both"
echo "  Seeds: 123, 456, 789"
echo "  Total models: $((${#SEEDS[@]} * ${#SECTIONS[@]}))"
echo ""

model_count=0
for section in "${SECTIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        model_count=$((model_count + 1))
        
        echo "Training BC model $model_count/9: section=$section, seed=$seed"
        
        mamba run -n main python -m passive_walker.bc.training.train \
            --backend torch \
            --section "$section" \
            --data "$FSM_DATA_DIR" \
            --epochs $EPOCHS \
            --batch 512 \
            --lr 1e-3 \
            --seed $seed \
            --save-dir "$BC_DIR" \
            --gpu
        
        echo ""
    done
done

echo ""
echo "  All BC models saved to: $BC_DIR"
echo ""

# =============================================================================
# 3. EVALUATE AND COMPARE BC MODELS
# =============================================================================
echo ""
echo "========================================================================"
echo "  STEP 3: EVALUATING AND COMPARING BC MODELS"
echo "========================================================================"
echo ""

BC_EVAL_DIR="$OUTPUT_DIR/bc_evaluation"
mkdir -p "$BC_EVAL_DIR"

# Create Python script for evaluation and comparison
cat > "$BC_EVAL_DIR/evaluate_all_models.py" << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.utils import Normalizer
from passive_walker.bc.models.models_torch import TorchMLPLarge
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse

def evaluate_model(model_path, meta_path, n_episodes=10):
    """Evaluate a single BC model."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    # Create model
    model = TorchMLPLarge(
        in_dim=meta['input_dim'],
        out_dim=meta['output_dim'],
        hidden=512
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create environment
    env = PassiveWalkerEnv(mode='research')
    normalizer = Normalizer(
        mean=np.array(checkpoint['normalizer_mean']),
        std=np.array(checkpoint['normalizer_std'])
    )
    
    # Run episodes
    returns = []
    lengths = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        while env.data.time < 25.0:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            obs_norm = normalizer.encode(obs_tensor)
            with torch.no_grad():
                action = model(obs_norm).squeeze(0).numpy()
            
            obs, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
            
            if done:
                break
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    env.close()
    
    return {
        'model_name': Path(model_path).stem,
        'returns': returns,
        'lengths': lengths,
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'success_rate': np.mean([l > 100 for l in lengths]),
        'section': meta['section'],
        'seed': meta['seed']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_episodes', type=int, default=10)
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all models
    model_files = list(models_dir.glob('*.pt'))
    
    print(f"Found {len(model_files)} models to evaluate")
    
    all_results = []
    
    for model_path in model_files:
        meta_path = model_path.with_suffix('.json')
        
        if not meta_path.exists():
            print(f"Skipping {model_path.name} (no metadata)")
            continue
        
        print(f"Evaluating {model_path.name}...")
        result = evaluate_model(str(model_path), str(meta_path), args.n_episodes)
        all_results.append(result)
        
        print(f"  Avg Return: {result['avg_return']:.2f} ± {result['std_return']:.2f}")
        print(f"  Success Rate: {result['success_rate']:.1%}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BC Models Comparison', fontsize=16, fontweight='bold')
    
    # Group by section
    sections = {}
    for r in all_results:
        section = r['section']
        if section not in sections:
            sections[section] = []
        sections[section].append(r)
    
    # 1. Average return by section
    ax = axes[0, 0]
    sections_order = ['hip', 'knees', 'both']
    for section in sections_order:
        if section in sections:
            values = [r['avg_return'] for r in sections[section]]
            ax.bar(section, np.mean(values), yerr=np.std(values), capsize=5, alpha=0.7)
    ax.set_title('Average Return by Section')
    ax.set_ylabel('Average Return')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Success rate by section
    ax = axes[0, 1]
    for section in sections_order:
        if section in sections:
            values = [r['success_rate'] for r in sections[section]]
            ax.bar(section, np.mean(values), yerr=np.std(values), capsize=5, alpha=0.7)
    ax.set_title('Success Rate by Section')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Return comparison (all models)
    ax = axes[1, 0]
    model_names = [r['model_name'] for r in all_results]
    returns = [r['avg_return'] for r in all_results]
    ax.barh(model_names, returns, alpha=0.7)
    ax.set_title('Average Return - All Models')
    ax.set_xlabel('Average Return')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Episode length by section
    ax = axes[1, 1]
    for section in sections_order:
        if section in sections:
            lengths = []
            for r in sections[section]:
                lengths.extend(r['lengths'])
            ax.hist(lengths, alpha=0.5, label=section, bins=20)
    ax.set_title('Episode Length Distribution')
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bc_models_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_dir / 'bc_models_comparison.png'}")
    plt.close()
    
    # Save summary metrics
    summary = {
        'total_models': len(all_results),
        'models': [
            {
                'name': r['model_name'],
                'section': r['section'],
                'seed': r['seed'],
                'avg_return': float(r['avg_return']),
                'success_rate': float(r['success_rate']),
                'avg_length': float(r['avg_length'])
            }
            for r in all_results
        ]
    }
    
    with open(output_dir / 'bc_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nEvaluation complete!")

if __name__ == '__main__':
    main()
EOF

# Run evaluation
echo "Running BC model evaluation and comparison..."
mamba run -n main python "$BC_EVAL_DIR/evaluate_all_models.py" \
    --models_dir "$BC_DIR" \
    --output_dir "$BC_EVAL_DIR" \
    --n_episodes 10

echo ""
echo "  BC evaluation complete. Results saved to: $BC_EVAL_DIR"
echo ""

# =============================================================================
# 4. TRAIN PPO MODELS (SWEEP)
# =============================================================================
echo ""
echo "========================================================================"
echo "  STEP 4: TRAINING PPO MODELS (SWEEP)"
echo "========================================================================"
echo ""

PPO_DIR="$OUTPUT_DIR/ppo_models"
mkdir -p "$PPO_DIR"

# PPO Sweep Configuration
PPO_SEEDS=(42 123 456)
TIMESTEPS=500000

echo "Training PPO models with different seeds..."
echo "  Seeds: ${PPO_SEEDS[@]}"
echo "  Timesteps: $TIMESTEPS"
echo ""

ppo_count=0
for seed in "${PPO_SEEDS[@]}"; do
    ppo_count=$((ppo_count + 1))
    
    echo "Training PPO model $ppo_count/3: seed=$seed"
    
    mamba run -n main python -m passive_walker.ppo.train \
        --experiment_name "ppo_overnight_seed${seed}" \
        --model_type mlp \
        --hidden_sizes 64 64 \
        --timesteps $TIMESTEPS \
        --eval_freq 25000 \
        --learning_rate 3e-4 \
        --n_steps 2048 \
        --batch_size 64 \
        --seed $seed \
        --device cpu \
        --out "$PPO_DIR"
    
    echo ""
done

echo ""
echo "  All PPO models saved to: $PPO_DIR"
echo ""

# =============================================================================
# 5. EVALUATE AND COMPARE PPO MODELS
# =============================================================================
echo ""
echo "========================================================================"
echo "  STEP 5: EVALUATING AND COMPARING PPO MODELS"
echo "========================================================================"
echo ""

PPO_EVAL_DIR="$OUTPUT_DIR/ppo_evaluation"
mkdir -p "$PPO_EVAL_DIR"

# Plot results for each PPO run
echo "Generating plots for each PPO run..."

for seed in "${PPO_SEEDS[@]}"; do
    run_dir="$PPO_DIR/ppo_overnight_seed${seed}"
    
    if [ -d "$run_dir" ]; then
        echo "Plotting results for seed $seed..."
        
        mamba run -n main python -m passive_walker.ppo.plot_ppo_results \
            --logdir "$run_dir" \
            --output "$PPO_EVAL_DIR/seed${seed}"
        
        echo ""
    fi
done

# Create comparison plot
echo "Creating PPO comparison plot..."

mamba run -n main python -c "
import sys
sys.path.insert(0, '$(pwd)')

import matplotlib.pyplot as plt
from pathlib import Path
import os

# Find all run directories
ppo_dir = Path('$PPO_DIR')
run_dirs = [d for d in ppo_dir.iterdir() if d.is_dir() and 'ppo_overnight_seed' in d.name]

if len(run_dirs) >= 3:
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PPO Models Comparison', fontsize=16, fontweight='bold')
    
    # For each seed, find the final metrics
    seeds = []
    final_returns = []
    final_lengths = []
    
    for seed in [42, 123, 456]:
        run_dir = ppo_dir / f'ppo_overnight_seed{seed}'
        
        # Look for tensorboard logs or metrics files
        # This is a simplified version - in practice you'd parse tensorboard logs
        seeds.append(f'Seed {seed}')
        # Placeholder values - in practice parse from actual logs
        final_returns.append(0)
        final_lengths.append(0)
    
    # Plot comparison
    ax = axes[0, 0]
    ax.bar(seeds, final_returns, alpha=0.7)
    ax.set_title('Final Return (placeholder)')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[0, 1]
    ax.bar(seeds, final_lengths, alpha=0.7)
    ax.set_title('Final Episode Length (placeholder)')
    ax.set_ylabel('Length (steps)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add text summary
    ax = axes[1, 0]
    ax.text(0.5, 0.5, 'PPO Evaluation plots\nsaved to respective\nseed directories', 
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'Trained {len(run_dirs)} PPO models\nwith seeds: 42, 123, 456', 
            ha='center', va='center', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('$PPO_EVAL_DIR/ppo_models_comparison.png', dpi=150, bbox_inches='tight')
    print('PPO comparison plot saved')
    plt.close()

print('PPO evaluation complete')
"

echo ""
echo "  PPO evaluation complete. Results saved to: $PPO_EVAL_DIR"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "========================================================================"
echo "  OVERNIGHT PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── fsm_data/              # FSM collected data"
echo "    ├── bc_models/             # BC trained models"
echo "    ├── bc_evaluation/         # BC evaluation plots"
echo "    ├── ppo_models/            # PPO trained models"
echo "    └── ppo_evaluation/        # PPO evaluation plots"
echo ""
echo "To view results:"
echo "  1. Check FSM visualization: $OUTPUT_DIR/fsm_data_visualization.png"
echo "  2. Check BC evaluations: $BC_EVAL_DIR/*.png"
echo "  3. Check PPO plots: $PPO_EVAL_DIR/*.png"
echo ""
