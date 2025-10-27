"""
PPO Evaluation and Comparison Tools

Comprehensive evaluation and comparison of PPO policies with BC and FSM baselines.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from passive_walker.config.paths import PPO_PLOTS_DIR, METRICS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir
import json
import time

from .models import create_actor_critic
from .config import PPOConfig
from ..core.env import PassiveWalkerEnv
from ..core.randomization import get_randomization_config
from ..bc.models.models_torch import TorchMLP
from ..bc.models.temporal_torch import TorchLSTM, TorchGRU


class PolicyEvaluator:
    """
    Comprehensive policy evaluator for PPO, BC, and FSM policies.
    
    Evaluates policies across multiple physics conditions and metrics.
    """
    
    def __init__(self, 
                 env_kwargs: Dict[str, Any] = None,
                 n_eval_episodes: int = 10,
                 deterministic: bool = True):
        """
        Initialize policy evaluator.
        
        Args:
            env_kwargs: Environment configuration
            n_eval_episodes: Number of episodes per evaluation
            deterministic: Whether to use deterministic actions
        """
        self.env_kwargs = env_kwargs or {}
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        
        # Evaluation results storage
        self.results = {}
    
    def evaluate_policy(self, 
                       policy: Any,
                       policy_name: str,
                       physics_conditions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate a policy across multiple physics conditions.
        
        Args:
            policy: Policy to evaluate (PPO model, BC model, or "fsm")
            policy_name: Name of the policy
            physics_conditions: List of physics conditions to test
            
        Returns:
            Dictionary of evaluation results
        """
        if physics_conditions is None:
            physics_conditions = [
                {"name": "baseline", "ramp_deg": 10.0, "friction": 0.9},
                {"name": "steep", "ramp_deg": 15.0, "friction": 0.9},
                {"name": "slippery", "ramp_deg": 10.0, "friction": 0.6},
                {"name": "rough", "ramp_deg": 10.0, "friction": 1.2},
                {"name": "randomized", "randomize_physics": True}
            ]
        
        results = {
            "policy_name": policy_name,
            "physics_conditions": {},
            "overall_stats": {}
        }
        
        for condition in physics_conditions:
            condition_name = condition["name"]
            print(f"Evaluating {policy_name} on {condition_name} condition...")
            
            # Create environment with specific physics
            env_kwargs = self.env_kwargs.copy()
            condition_copy = condition.copy()
            condition_copy.pop("name", None)  # Remove name key
            env_kwargs.update(condition_copy)
            
            env = PassiveWalkerEnv(**env_kwargs)
            
            # Evaluate policy
            condition_results = self._evaluate_single_condition(policy, env)
            results["physics_conditions"][condition_name] = condition_results
        
        # Compute overall statistics
        all_returns = []
        all_lengths = []
        all_success_rates = []
        
        for condition_results in results["physics_conditions"].values():
            all_returns.extend(condition_results["returns"])
            all_lengths.extend(condition_results["lengths"])
            all_success_rates.append(condition_results["success_rate"])
        
        results["overall_stats"] = {
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "mean_length": np.mean(all_lengths),
            "std_length": np.std(all_lengths),
            "mean_success_rate": np.mean(all_success_rates),
            "std_success_rate": np.std(all_success_rates)
        }
        
        self.results[policy_name] = results
        return results
    
    def _evaluate_single_condition(self, policy: Any, env: PassiveWalkerEnv) -> Dict[str, Any]:
        """
        Evaluate policy on a single physics condition.
        
        Args:
            policy: Policy to evaluate
            env: Environment with specific physics
            
        Returns:
            Dictionary of evaluation results
        """
        # CRITICAL FIX: Remove FSM fallback - force error if policy is string
        if isinstance(policy, str):
            raise ValueError(f"Cannot evaluate with string policy: {policy}")
        
        # Validate policy has required method
        if not hasattr(policy, 'get_action'):
            raise ValueError(f"Policy {type(policy)} missing get_action method")
        
        # Log what we're evaluating
        print(f"  Evaluating policy type: {type(policy).__name__}")
        
        returns = []
        lengths = []
        success_count = 0
        
        for episode in range(self.n_eval_episodes):
            obs, _ = env.reset()
            episode_return = 0
            episode_length = 0
            
            while True:
                # Get action from policy
                if hasattr(policy, 'get_action'):
                    # PPO model
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        action_output = policy.get_action(obs_tensor, deterministic=self.deterministic)
                        if len(action_output) == 4:
                            # Temporal model
                            action, _, _, _ = action_output
                        else:
                            # MLP model
                            action, _, _ = action_output
                        action = action.squeeze(0).cpu().numpy()
                        
                        # Add action validation
                        assert not np.isnan(action).any(), f"NaN in action: {action}"
                        assert not np.isinf(action).any(), f"Inf in action: {action}"
                        assert (np.abs(action) < 10).all(), f"Action out of range: {action}"
                else:
                    # BC model
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                        if hasattr(policy, 'forward') and 'hidden' in policy.forward.__code__.co_varnames:
                            # Temporal model
                            prediction, _ = policy(obs_tensor)
                        else:
                            # MLP model
                            prediction = policy(obs_tensor)
                        action = prediction.squeeze(0).cpu().numpy()
                        
                        # Add action validation for BC models too
                        assert not np.isnan(action).any(), f"NaN in action: {action}"
                        assert not np.isinf(action).any(), f"Inf in action: {action}"
                        assert (np.abs(action) < 10).all(), f"Action out of range: {action}"
                
                # Take action
                obs, reward, done, info = env.step(action)
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
            
            returns.append(episode_return)
            lengths.append(episode_length)
            
            # Determine success (episode length > 50 and positive return)
            if episode_length > 50 and episode_return > 0:
                success_count += 1
        
        success_rate = success_count / self.n_eval_episodes
        
        return {
            "returns": returns,
            "lengths": lengths,
            "success_rate": success_rate,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths)
        }
    
    def compare_policies(self, policy_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple policies across all conditions.
        
        Args:
            policy_names: Names of policies to compare (if None, uses all evaluated)
            
        Returns:
            Dictionary of comparison results
        """
        if policy_names is None:
            policy_names = list(self.results.keys())
        
        comparison = {
            "policies": policy_names,
            "overall_comparison": {},
            "condition_comparison": {}
        }
        
        # Overall comparison
        for policy_name in policy_names:
            if policy_name in self.results:
                comparison["overall_comparison"][policy_name] = self.results[policy_name]["overall_stats"]
        
        # Condition-specific comparison
        if policy_names:
            first_policy = policy_names[0]
            if first_policy in self.results:
                for condition_name in self.results[first_policy]["physics_conditions"]:
                    comparison["condition_comparison"][condition_name] = {}
                    
                    for policy_name in policy_names:
                        if policy_name in self.results:
                            condition_results = self.results[policy_name]["physics_conditions"][condition_name]
                            comparison["condition_comparison"][condition_name][policy_name] = {
                                "mean_return": condition_results["mean_return"],
                                "std_return": condition_results["std_return"],
                                "success_rate": condition_results["success_rate"]
                            }
        
        return comparison
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filepath: str):
        """Load evaluation results from JSON file."""
        with open(filepath, 'r') as f:
            self.results = json.load(f)


class PolicyVisualizer:
    """
    Policy visualization and comparison tools.
    
    Creates plots for policy comparison, learning curves, and performance analysis.
    """
    
    def __init__(self, save_dir: str = str(PPO_PLOTS_DIR)):
        """
        Initialize policy visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        redirected = redirect_legacy_dir(save_dir)
        self.save_dir = Path(redirected)
        ensure_dir_exists(self.save_dir)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_policy_comparison(self, evaluator: PolicyEvaluator, 
                              policy_names: List[str] = None,
                              save_name: str = "policy_comparison"):
        """
        Create policy comparison plots.
        
        Args:
            evaluator: PolicyEvaluator with results
            policy_names: Names of policies to compare
            save_name: Name for saved plot file
        """
        if policy_names is None:
            policy_names = list(evaluator.results.keys())
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Policy Comparison', fontsize=16)
        
        # 1. Overall performance comparison
        ax1 = axes[0, 0]
        policies = []
        mean_returns = []
        std_returns = []
        
        for policy_name in policy_names:
            if policy_name in evaluator.results:
                stats = evaluator.results[policy_name]["overall_stats"]
                policies.append(policy_name)
                mean_returns.append(stats["mean_return"])
                std_returns.append(stats["std_return"])
        
        ax1.bar(policies, mean_returns, yerr=std_returns, capsize=5)
        ax1.set_title('Overall Performance')
        ax1.set_ylabel('Mean Return')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Success rate comparison
        ax2 = axes[0, 1]
        success_rates = []
        
        for policy_name in policy_names:
            if policy_name in evaluator.results:
                stats = evaluator.results[policy_name]["overall_stats"]
                success_rates.append(stats["mean_success_rate"])
        
        ax2.bar(policies, success_rates)
        ax2.set_title('Success Rate')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance across physics conditions
        ax3 = axes[1, 0]
        
        if policy_names and policy_names[0] in evaluator.results:
            condition_names = list(evaluator.results[policy_names[0]]["physics_conditions"].keys())
            
            for policy_name in policy_names:
                if policy_name in evaluator.results:
                    condition_returns = []
                    for condition_name in condition_names:
                        condition_results = evaluator.results[policy_name]["physics_conditions"][condition_name]
                        condition_returns.append(condition_results["mean_return"])
                    
                    ax3.plot(condition_names, condition_returns, marker='o', label=policy_name)
            
            ax3.set_title('Performance Across Physics Conditions')
            ax3.set_ylabel('Mean Return')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
        
        # 4. Episode length comparison
        ax4 = axes[1, 1]
        mean_lengths = []
        std_lengths = []
        
        for policy_name in policy_names:
            if policy_name in evaluator.results:
                stats = evaluator.results[policy_name]["overall_stats"]
                mean_lengths.append(stats["mean_length"])
                std_lengths.append(stats["std_length"])
        
        ax4.bar(policies, mean_lengths, yerr=std_lengths, capsize=5)
        ax4.set_title('Episode Length')
        ax4.set_ylabel('Mean Length')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_learning_curves(self, training_logs: Dict[str, List[float]], 
                            save_name: str = "learning_curves"):
        """
        Plot learning curves for training metrics.
        
        Args:
            training_logs: Dictionary of metric names to lists of values
            save_name: Name for saved plot file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Learning Curves', fontsize=16)
        
        metrics = [
            ("policy_loss", "Policy Loss"),
            ("value_loss", "Value Loss"),
            ("entropy_loss", "Entropy Loss"),
            ("kl_divergence", "KL Divergence")
        ]
        
        for i, (metric_key, metric_title) in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            if metric_key in training_logs:
                values = training_logs[metric_key]
                ax.plot(values)
                ax.set_title(metric_title)
                ax.set_xlabel('Update')
                ax.set_ylabel(metric_title)
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_action_distributions(self, policy: nn.Module, 
                                 env: PassiveWalkerEnv,
                                 n_samples: int = 1000,
                                 save_name: str = "action_distributions"):
        """
        Plot action distributions from policy.
        
        Args:
            policy: Policy to analyze
            env: Environment for sampling
            n_samples: Number of action samples
            save_name: Name for saved plot file
        """
        actions = []
        
        for _ in range(n_samples):
            obs, _ = env.reset()
            
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                if hasattr(policy, 'get_action'):
                    # PPO model
                    action_output = policy.get_action(obs_tensor, deterministic=False)
                    if len(action_output) == 4:
                        # Temporal model
                        action, _, _, _ = action_output
                    else:
                        # MLP model
                        action, _, _ = action_output
                else:
                    # BC model
                    if hasattr(policy, 'forward') and 'hidden' in policy.forward.__code__.co_varnames:
                        # Temporal model
                        prediction, _ = policy(obs_tensor)
                    else:
                        # MLP model
                        prediction = policy(obs_tensor)
                    action = prediction
                
                action = action.squeeze(0).cpu().numpy()
                actions.append(action)
        
        actions = np.array(actions)
        
        # Create action distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Action Distributions', fontsize=16)
        
        action_names = ['Hip', 'Left Knee', 'Right Knee']
        
        for i in range(3):
            ax = axes[i]
            ax.hist(actions[:, i], bins=50, alpha=0.7, density=True)
            ax.set_title(f'{action_names[i]} Action')
            ax.set_xlabel('Action Value')
            ax.set_ylabel('Density')
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_policy_evaluator(**kwargs) -> PolicyEvaluator:
    """Create policy evaluator with default settings."""
    return PolicyEvaluator(**kwargs)


def create_policy_visualizer(save_dir: str = str(PPO_PLOTS_DIR)) -> PolicyVisualizer:
    """Create policy visualizer with default settings."""
    return PolicyVisualizer(save_dir)
