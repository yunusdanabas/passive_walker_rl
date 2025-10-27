#!/usr/bin/env python3
"""
Simple Model Comparison Tool

Compare multiple models side by side.
"""

import argparse
import json
from pathlib import Path
import sys

from passive_walker.config.paths import REPORTS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir


def load_metrics(model_path: str):
    """Load metrics from a model directory."""
    metrics_file = Path(model_path).parent / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None


def compare_bc_models(model_paths: list, episodes: int = 10):
    """Compare multiple BC models."""
    try:
        from passive_walker.bc.evaluation.evaluate import evaluate_bc_model
        
        results = {}
        for model_path in model_paths:
            print(f"Evaluating {model_path}...")
            metrics = evaluate_bc_model(model_path, episodes, gui=False)
            if metrics:
                results[Path(model_path).stem] = metrics
        
        # Print comparison
        print("\n" + "="*80)
        print("Model Comparison Results")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
            print(f"  Avg episode length: {metrics.get('avg_episode_length', 0):.2f}")
            print(f"  Avg reward: {metrics.get('avg_reward', 0):.2f}")
        
        return results
    except Exception as e:
        print(f"Error comparing models: {e}")
        return None


def compare_ppo_models(model_paths: list, episodes: int = 10):
    """Compare multiple PPO models."""
    try:
        from passive_walker.ppo.evaluate import evaluate_ppo
        
        results = {}
        for model_path in model_paths:
            print(f"Evaluating {model_path}...")
            metrics = evaluate_ppo(model_path, episodes, gui=False)
            if metrics:
                results[Path(model_path).stem] = metrics
        
        # Print comparison
        print("\n" + "="*80)
        print("Model Comparison Results")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"  Avg return: {metrics.get('avg_return', 0):.2f}")
            print(f"  Avg episode length: {metrics.get('avg_episode_length', 0):.2f}")
            print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
        
        return results
    except Exception as e:
        print(f"Error comparing models: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models")
    parser.add_argument("models", nargs="+", help="Model checkpoint paths")
    parser.add_argument("--type", type=str, choices=["bc", "ppo"], required=True,
                       help="Model type: bc or ppo")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes per model")
    parser.add_argument("--output", type=str, default=str(REPORTS_DIR / "model_comparison.md"),
                    help="Output markdown report path")
    
    args = parser.parse_args()
    
    # Redirect and ensure output directory exists
    out_path = Path(redirect_legacy_dir(args.output))
    ensure_dir_exists(out_path.parent)

    # Compare models
    if args.type == 'bc':
        results = compare_bc_models(args.models, args.episodes)
    elif args.type == 'ppo':
        results = compare_ppo_models(args.models, args.episodes)
    else:
        print(f"Unknown model type: {args.type}")
        return
    
    # Save results if requested
    if args.out and results:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nComparison results saved to: {args.out}")


if __name__ == "__main__":
    main()

