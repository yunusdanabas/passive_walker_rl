#!/usr/bin/env python3
"""
Simple Model Evaluation Tool

Evaluate BC or PPO models with basic metrics and optional visualization.
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.config.paths import METRICS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir


def evaluate_bc_model(model_path: str, episodes: int = 10, gui: bool = False):
    """Evaluate a BC model."""
    try:
        from passive_walker.bc.evaluation.evaluate import evaluate_bc_model
        
        print(f"Evaluating BC model: {model_path}")
        print(f"Episodes: {episodes}")
        
        metrics = evaluate_bc_model(
            model_path=model_path,
            episodes=episodes,
            gui=gui
        )
        
        print("\nEvaluation Results:")
        print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
        print(f"  Avg episode length: {metrics.get('avg_episode_length', 0):.2f}")
        print(f"  Avg reward: {metrics.get('avg_reward', 0):.2f}")
        
        return metrics
    except ImportError as e:
        print(f"Error importing BC evaluation: {e}")
        return None


def evaluate_ppo_model(model_path: str, episodes: int = 10, gui: bool = False):
    """Evaluate a PPO model."""
    try:
        from passive_walker.ppo.evaluate import evaluate_ppo
        
        print(f"Evaluating PPO model: {model_path}")
        print(f"Episodes: {episodes}")
        
        metrics = evaluate_ppo(
            model_path=model_path,
            episodes=episodes,
            gui=gui
        )
        
        print("\nEvaluation Results:")
        print(f"  Avg return: {metrics.get('avg_return', 0):.2f}")
        print(f"  Avg episode length: {metrics.get('avg_episode_length', 0):.2f}")
        print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
        
        return metrics
    except Exception as e:
        print(f"Error evaluating PPO model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC or PPO model")
    parser.add_argument("model", type=str, help="Path to model file")
    parser.add_argument("--type", type=str, choices=["bc", "ppo"], required=True,
                       help="Model type: bc or ppo")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                       help="Show GUI during evaluation")
    parser.add_argument("--out", type=str, default=None,
                       help="Output file for metrics")
    
    args = parser.parse_args()
    
    # Determine model type if not specified
    if args.type is None:
        if args.model.endswith('.pt'):
            args.type = 'bc'
        elif args.model.endswith('.pth'):
            args.type = 'ppo'
        else:
            print("Error: Cannot determine model type. Please specify --type")
            return
    
    # Evaluate model
    if args.type == 'bc':
        metrics = evaluate_bc_model(args.model, args.episodes, args.gui)
    elif args.type == 'ppo':
        metrics = evaluate_ppo_model(args.model, args.episodes, args.gui)
    else:
        print(f"Unknown model type: {args.type}")
        return
    
    # Save metrics if requested
    if args.out and metrics:
        out_path = Path(redirect_legacy_dir(args.out))
        ensure_dir_exists(out_path.parent)
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()

