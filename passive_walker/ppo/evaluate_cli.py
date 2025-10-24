"""
PPO Evaluation CLI

Command-line interface for evaluating PPO policies and comparing with BC/FSM baselines.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from passive_walker.ppo.models import create_actor_critic
from passive_walker.ppo.config import PPOConfig
from passive_walker.ppo.evaluate import PolicyEvaluator, PolicyVisualizer
from passive_walker.bc.models.models_torch import TorchMLP
import torch
import json


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="PPO policy evaluation")
    
    # Basic arguments
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                       help="Number of episodes per evaluation")
    parser.add_argument("--deterministic", action="store_true",
                       help="Use deterministic actions")
    parser.add_argument("--out", type=str, default="evaluation_results",
                       help="Output directory")
    
    # Policy arguments
    parser.add_argument("--ppo_model", type=str, default=None,
                       help="Path to PPO model checkpoint")
    parser.add_argument("--bc_model", type=str, default=None,
                       help="Path to BC model checkpoint")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare PPO, BC, and FSM policies")
    
    # Evaluation arguments
    parser.add_argument("--physics_conditions", type=str, nargs="+", 
                       default=["baseline", "steep", "slippery", "rough", "randomized"],
                       help="Physics conditions to test")
    parser.add_argument("--save_plots", action="store_true",
                       help="Save evaluation plots")
    
    args = parser.parse_args()
    
    print(f"Starting policy evaluation...")
    print(f"Episodes per condition: {args.n_eval_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Physics conditions: {args.physics_conditions}")
    
    # Create evaluator
    evaluator = PolicyEvaluator(
        n_eval_episodes=args.n_eval_episodes,
        deterministic=args.deterministic
    )
    
    # Create visualizer
    visualizer = PolicyVisualizer(save_dir=args.out)
    
    # Evaluate policies
    policies_evaluated = []
    
    if args.compare_all or args.ppo_model:
        # Evaluate PPO policy
        if args.ppo_model:
            # Load PPO model
            checkpoint = torch.load(args.ppo_model, map_location="cpu")
            model_config = checkpoint.get("config", {})
            
            if model_config.get("model_type") == "mlp":
                model = create_actor_critic(
                    "mlp",
                    obs_dim=17,
                    action_dim=3,
                    hidden_sizes=model_config.get("hidden_sizes", [64, 64])
                )
            else:
                model = create_actor_critic(
                    model_config.get("model_type", "lstm"),
                    obs_dim=17,
                    action_dim=3,
                    hidden_size=model_config.get("hidden_size", 64),
                    num_layers=model_config.get("num_layers", 1)
                )
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            print("Evaluating PPO model...")
            ppo_results = evaluator.evaluate_policy(model, "ppo_model")
            policies_evaluated.append("ppo_model")
        else:
            # Create dummy PPO model for testing
            model = create_actor_critic("mlp", obs_dim=17, action_dim=3, hidden_sizes=[32, 32])
            print("Evaluating dummy PPO model...")
            ppo_results = evaluator.evaluate_policy(model, "ppo_dummy")
            policies_evaluated.append("ppo_dummy")
    
    if args.compare_all or args.bc_model:
        # Evaluate BC policy
        if args.bc_model:
            # Load BC model
            checkpoint = torch.load(args.bc_model, map_location="cpu")
            model_config = checkpoint.get("config", {})
            
            model = TorchMLP(
                in_dim=17,
                out_dim=3,
                hidden=model_config.get("hidden_sizes", [64, 64])
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            
            print("Evaluating BC model...")
            bc_results = evaluator.evaluate_policy(model, "bc_model")
            policies_evaluated.append("bc_model")
        else:
            # Create dummy BC model for testing (skip for now due to BC model issues)
            print("Skipping BC model evaluation due to model creation issues")
            # model = TorchMLP(in_dim=17, out_dim=3, hidden=[32, 32])
            # print("Evaluating dummy BC model...")
            # bc_results = evaluator.evaluate_policy(model, "bc_dummy")
            # policies_evaluated.append("bc_dummy")
    
    if args.compare_all:
        # Evaluate FSM policy
        print("Evaluating FSM baseline...")
        fsm_results = evaluator.evaluate_policy("fsm", "fsm_baseline")
        policies_evaluated.append("fsm_baseline")
    
    # Compare policies
    if len(policies_evaluated) > 1:
        print(f"\nComparing policies: {policies_evaluated}")
        comparison = evaluator.compare_policies(policies_evaluated)
        
        # Print comparison results
        print("\n=== Policy Comparison ===")
        for policy_name in policies_evaluated:
            if policy_name in comparison["overall_comparison"]:
                stats = comparison["overall_comparison"][policy_name]
                print(f"\n{policy_name}:")
                print(f"  Mean return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
                print(f"  Mean length: {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
                print(f"  Success rate: {stats['mean_success_rate']:.2f} ± {stats['std_success_rate']:.2f}")
        
        # Save comparison results
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "comparison_results.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison results saved to: {args.out}/comparison_results.json")
        
        # Create plots
        if args.save_plots:
            print("Creating comparison plots...")
            visualizer.plot_policy_comparison(evaluator, policies_evaluated)
            print(f"Plots saved to: {args.out}/")
    
    # Save individual results
    os.makedirs(args.out, exist_ok=True)
    evaluator.save_results(os.path.join(args.out, "evaluation_results.json"))
    print(f"Evaluation results saved to: {args.out}/evaluation_results.json")
    
    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
