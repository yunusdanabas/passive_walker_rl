#!/usr/bin/env python3
"""
Robustness Testing Suite for Passive Walker Models

This module implements comprehensive robustness testing across various conditions:
- Control frequency variations (50Hz, 100Hz, 150Hz, 200Hz)
- Observation noise injection (0%, 1%, 5%, 10%)
- Action noise injection (0%, 1%, 5%, 10%)
- Physics parameter sweeps (gravity, mass, friction, damping)
- Missing data scenarios (observation dropout)
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.physics_conditions import PhysicsConditionManager, PhysicsParameter
from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
from passive_walker.bc.models.models_jax import MLP
from passive_walker.bc.models.temporal_torch import TorchLSTM, TorchGRU, TorchBiLSTM
from passive_walker.bc.models.temporal_jax import LSTM, GRU


@dataclass
class RobustnessConfig:
    """Configuration for robustness testing."""
    # Control frequency testing
    control_frequencies: List[float] = None  # Hz
    
    # Noise testing
    obs_noise_levels: List[float] = None  # Percentage (0.0 to 1.0)
    action_noise_levels: List[float] = None  # Percentage (0.0 to 1.0)
    
    # Physics parameter testing
    physics_params: List[PhysicsParameter] = None
    physics_ranges: Dict[PhysicsParameter, Tuple[float, float]] = None
    
    # Missing data testing
    dropout_rates: List[float] = None  # Percentage of observations to drop
    
    # Evaluation parameters
    episodes_per_condition: int = 10
    max_steps_per_episode: int = 1000
    dt: float = 0.01
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.control_frequencies is None:
            self.control_frequencies = [50.0, 100.0, 150.0, 200.0]
        
        if self.obs_noise_levels is None:
            self.obs_noise_levels = [0.0, 0.01, 0.05, 0.10]
        
        if self.action_noise_levels is None:
            self.action_noise_levels = [0.0, 0.01, 0.05, 0.10]
        
        if self.physics_params is None:
            self.physics_params = [
                PhysicsParameter.GRAVITY,
                PhysicsParameter.MASS,
                PhysicsParameter.FRICTION,
                PhysicsParameter.DAMPING
            ]
        
        if self.physics_ranges is None:
            self.physics_ranges = {
                PhysicsParameter.GRAVITY: (0.7, 1.3),  # 70% to 130% of Earth gravity
                PhysicsParameter.MASS: (0.8, 1.2),      # 80% to 120% of default mass
                PhysicsParameter.FRICTION: (0.5, 1.0),   # 50% to 100% friction
                PhysicsParameter.DAMPING: (0.3, 1.5)    # 30% to 150% damping
            }
        
        if self.dropout_rates is None:
            self.dropout_rates = [0.0, 0.05, 0.10, 0.20]


@dataclass
class RobustnessResult:
    """Results from robustness testing."""
    condition_name: str
    condition_params: Dict
    metrics: Dict[str, float]
    episodes: List[Dict]
    success_rate: float
    avg_distance: float
    avg_reward: float
    failure_modes: List[str]


class RobustnessTester:
    """Comprehensive robustness testing for passive walker models."""
    
    def __init__(self, config: Optional[RobustnessConfig] = None):
        """Initialize robustness tester.
        
        Args:
            config: Robustness testing configuration
        """
        self.config = config or RobustnessConfig()
        self.results: List[RobustnessResult] = []
        self.physics_manager = PhysicsConditionManager()
        
    def test_model_robustness(self, 
                            model_path: str,
                            backend: str = "torch",
                            model_type: str = "mlp") -> Dict[str, RobustnessResult]:
        """Test model robustness across all conditions.
        
        Args:
            model_path: Path to model checkpoint
            backend: "torch" or "jax"
            model_type: "mlp", "lstm", "gru", "bilstm"
            
        Returns:
            Dictionary mapping condition names to results
        """
        print(f"Starting robustness testing for {model_type} model ({backend})")
        print(f"Model path: {model_path}")
        
        # Load model
        model = self._load_model(model_path, backend, model_type)
        
        # Test different robustness dimensions
        all_results = {}
        
        # 1. Control frequency testing
        print("\n=== Testing Control Frequencies ===")
        freq_results = self._test_control_frequencies(model, backend)
        all_results.update(freq_results)
        
        # 2. Observation noise testing
        print("\n=== Testing Observation Noise ===")
        obs_noise_results = self._test_observation_noise(model, backend)
        all_results.update(obs_noise_results)
        
        # 3. Action noise testing
        print("\n=== Testing Action Noise ===")
        action_noise_results = self._test_action_noise(model, backend)
        all_results.update(action_noise_results)
        
        # 4. Physics parameter testing
        print("\n=== Testing Physics Parameters ===")
        physics_results = self._test_physics_parameters(model, backend)
        all_results.update(physics_results)
        
        # 5. Missing data testing
        print("\n=== Testing Missing Data ===")
        dropout_results = self._test_missing_data(model, backend)
        all_results.update(dropout_results)
        
        # Store all results
        self.results.extend(all_results.values())
        
        print(f"\nRobustness testing completed: {len(all_results)} conditions tested")
        return all_results
    
    def _load_model(self, model_path: str, backend: str, model_type: str):
        """Load model from checkpoint."""
        if backend == "torch":
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if model_type == "mlp":
                model = TorchMLP(input_dim=17, hidden_dim=256, output_dim=3)
            elif model_type == "mlp_large":
                model = TorchMLPLarge(input_dim=17, output_dim=3)
            elif model_type == "lstm":
                model = TorchLSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "gru":
                model = TorchGRU(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "bilstm":
                model = TorchBiLSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
            
        elif backend == "jax":
            import jax
            import equinox as eqx
            
            if model_type == "mlp":
                model = MLP(input_dim=17, hidden_dim=256, output_dim=3)
            elif model_type == "lstm":
                model = LSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "gru":
                model = GRU(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = eqx.tree_deserialise_leaves(model_path, model)
            return model
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _test_control_frequencies(self, model, backend: str) -> Dict[str, RobustnessResult]:
        """Test model robustness across different control frequencies."""
        results = {}
        
        for freq in self.config.control_frequencies:
            print(f"  Testing control frequency: {freq} Hz")
            
            # Create environment with specific control frequency
            env = PassiveWalkerEnv()
            env.dt = 1.0 / freq  # Set timestep based on frequency
            
            # Run episodes
            episodes = self._run_episodes(env, model, backend, f"freq_{freq}Hz")
            
            # Compute metrics
            metrics = self._compute_metrics(episodes)
            
            result = RobustnessResult(
                condition_name=f"control_freq_{freq}Hz",
                condition_params={"control_frequency": freq},
                metrics=metrics,
                episodes=episodes,
                success_rate=metrics["success_rate"],
                avg_distance=metrics["avg_distance"],
                avg_reward=metrics["avg_reward"],
                failure_modes=metrics["failure_modes"]
            )
            
            results[result.condition_name] = result
            print(f"    Success rate: {result.success_rate:.2%}")
        
        return results
    
    def _test_observation_noise(self, model, backend: str) -> Dict[str, RobustnessResult]:
        """Test model robustness with observation noise."""
        results = {}
        
        for noise_level in self.config.obs_noise_levels:
            print(f"  Testing observation noise: {noise_level:.1%}")
            
            # Create environment
            env = PassiveWalkerEnv()
            
            # Run episodes with observation noise
            episodes = self._run_episodes(env, model, backend, f"obs_noise_{noise_level:.1%}", 
                                        obs_noise=noise_level)
            
            # Compute metrics
            metrics = self._compute_metrics(episodes)
            
            result = RobustnessResult(
                condition_name=f"obs_noise_{noise_level:.1%}",
                condition_params={"obs_noise": noise_level},
                metrics=metrics,
                episodes=episodes,
                success_rate=metrics["success_rate"],
                avg_distance=metrics["avg_distance"],
                avg_reward=metrics["avg_reward"],
                failure_modes=metrics["failure_modes"]
            )
            
            results[result.condition_name] = result
            print(f"    Success rate: {result.success_rate:.2%}")
        
        return results
    
    def _test_action_noise(self, model, backend: str) -> Dict[str, RobustnessResult]:
        """Test model robustness with action noise."""
        results = {}
        
        for noise_level in self.config.action_noise_levels:
            print(f"  Testing action noise: {noise_level:.1%}")
            
            # Create environment
            env = PassiveWalkerEnv()
            
            # Run episodes with action noise
            episodes = self._run_episodes(env, model, backend, f"action_noise_{noise_level:.1%}", 
                                        action_noise=noise_level)
            
            # Compute metrics
            metrics = self._compute_metrics(episodes)
            
            result = RobustnessResult(
                condition_name=f"action_noise_{noise_level:.1%}",
                condition_params={"action_noise": noise_level},
                metrics=metrics,
                episodes=episodes,
                success_rate=metrics["success_rate"],
                avg_distance=metrics["avg_distance"],
                avg_reward=metrics["avg_reward"],
                failure_modes=metrics["failure_modes"]
            )
            
            results[result.condition_name] = result
            print(f"    Success rate: {result.success_rate:.2%}")
        
        return results
    
    def _test_physics_parameters(self, model, backend: str) -> Dict[str, RobustnessResult]:
        """Test model robustness across physics parameter variations."""
        results = {}
        
        for param in self.config.physics_params:
            param_range = self.config.physics_ranges[param]
            min_val, max_val = param_range
            
            # Test minimum, nominal, and maximum values
            test_values = [min_val, 1.0, max_val]  # 1.0 is nominal
            
            for val in test_values:
                print(f"  Testing {param.value}: {val:.2f}")
                
                # Create environment
                env = PassiveWalkerEnv()
                
                # Apply physics condition
                condition = {param.value: val}
                self.physics_manager.apply_condition_to_env(env, condition)
                
                # Run episodes
                episodes = self._run_episodes(env, model, backend, f"{param.value}_{val:.2f}")
                
                # Compute metrics
                metrics = self._compute_metrics(episodes)
                
                result = RobustnessResult(
                    condition_name=f"{param.value}_{val:.2f}",
                    condition_params={param.value: val},
                    metrics=metrics,
                    episodes=episodes,
                    success_rate=metrics["success_rate"],
                    avg_distance=metrics["avg_distance"],
                    avg_reward=metrics["avg_reward"],
                    failure_modes=metrics["failure_modes"]
                )
                
                results[result.condition_name] = result
                print(f"    Success rate: {result.success_rate:.2%}")
        
        return results
    
    def _test_missing_data(self, model, backend: str) -> Dict[str, RobustnessResult]:
        """Test model robustness with missing observations."""
        results = {}
        
        for dropout_rate in self.config.dropout_rates:
            print(f"  Testing dropout rate: {dropout_rate:.1%}")
            
            # Create environment
            env = PassiveWalkerEnv()
            
            # Run episodes with missing data
            episodes = self._run_episodes(env, model, backend, f"dropout_{dropout_rate:.1%}", 
                                        dropout_rate=dropout_rate)
            
            # Compute metrics
            metrics = self._compute_metrics(episodes)
            
            result = RobustnessResult(
                condition_name=f"dropout_{dropout_rate:.1%}",
                condition_params={"dropout_rate": dropout_rate},
                metrics=metrics,
                episodes=episodes,
                success_rate=metrics["success_rate"],
                avg_distance=metrics["avg_distance"],
                avg_reward=metrics["avg_reward"],
                failure_modes=metrics["failure_modes"]
            )
            
            results[result.condition_name] = result
            print(f"    Success rate: {result.success_rate:.2%}")
        
        return results
    
    def _run_episodes(self, env, model, backend: str, condition_name: str,
                     obs_noise: float = 0.0, action_noise: float = 0.0, 
                     dropout_rate: float = 0.0) -> List[Dict]:
        """Run episodes with the given model and conditions."""
        episodes = []
        
        for episode_idx in range(self.config.episodes_per_condition):
            obs, info = env.reset()
            episode_data = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "done": False,
                "total_reward": 0.0,
                "distance": 0.0,
                "steps": 0,
                "failure_mode": None
            }
            
            step_count = 0
            while step_count < self.config.max_steps_per_episode:
                # Apply observation noise
                if obs_noise > 0:
                    obs = self._add_observation_noise(obs, obs_noise)
                
                # Apply dropout
                if dropout_rate > 0:
                    obs = self._apply_dropout(obs, dropout_rate)
                
                # Get action from model
                action = self._get_model_action(model, obs, backend)
                
                # Apply action noise
                if action_noise > 0:
                    action = self._add_action_noise(action, action_noise)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store episode data
                episode_data["observations"].append(obs.copy())
                episode_data["actions"].append(action.copy())
                episode_data["rewards"].append(reward)
                episode_data["total_reward"] += reward
                episode_data["steps"] += 1
                
                # Check for failure
                if terminated or truncated:
                    episode_data["done"] = True
                    episode_data["failure_mode"] = self._classify_failure(obs, info)
                    break
                
                obs = next_obs
                step_count += 1
            
            # Compute final distance
            episode_data["distance"] = obs[0]  # x position
            
            episodes.append(episode_data)
        
        return episodes
    
    def _get_model_action(self, model, obs: np.ndarray, backend: str) -> np.ndarray:
        """Get action from model."""
        if backend == "torch":
            import torch
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action_tensor = model(obs_tensor)
                action = action_tensor.squeeze(0).numpy()
            
            return np.clip(action, -1.0, 1.0)
            
        elif backend == "jax":
            import jax.numpy as jnp
            
            obs_jax = jnp.array(obs)
            action = model(obs_jax)
            return np.clip(action, -1.0, 1.0)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _add_observation_noise(self, obs: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to observations."""
        noise = np.random.normal(0, noise_level, obs.shape)
        return obs + noise
    
    def _add_action_noise(self, action: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to actions."""
        noise = np.random.normal(0, noise_level, action.shape)
        return np.clip(action + noise, -1.0, 1.0)
    
    def _apply_dropout(self, obs: np.ndarray, dropout_rate: float) -> np.ndarray:
        """Randomly set observations to zero (simulate missing data)."""
        mask = np.random.random(obs.shape) > dropout_rate
        return obs * mask
    
    def _classify_failure(self, obs: np.ndarray, info: Dict) -> str:
        """Classify the type of failure."""
        pitch = obs[2]  # Pitch angle
        
        if pitch > 0.5:  # Forward fall
            return "forward_fall"
        elif pitch < -0.5:  # Backward fall
            return "backward_fall"
        elif abs(obs[0]) < 0.1:  # Stagnation
            return "stagnation"
        else:
            return "other"
    
    def _compute_metrics(self, episodes: List[Dict]) -> Dict[str, float]:
        """Compute metrics from episode data."""
        if not episodes:
            return {
                "success_rate": 0.0,
                "avg_distance": 0.0,
                "avg_reward": 0.0,
                "avg_steps": 0.0,
                "failure_modes": []
            }
        
        # Success rate (episodes that completed without failure)
        successful_episodes = [ep for ep in episodes if ep["failure_mode"] is None]
        success_rate = len(successful_episodes) / len(episodes)
        
        # Average distance
        avg_distance = np.mean([ep["distance"] for ep in episodes])
        
        # Average reward
        avg_reward = np.mean([ep["total_reward"] for ep in episodes])
        
        # Average steps
        avg_steps = np.mean([ep["steps"] for ep in episodes])
        
        # Failure modes
        failure_modes = [ep["failure_mode"] for ep in episodes if ep["failure_mode"] is not None]
        
        return {
            "success_rate": success_rate,
            "avg_distance": avg_distance,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "failure_modes": failure_modes
        }
    
    def generate_robustness_report(self, output_dir: str = "experiments/outputs/robustness"):
        """Generate comprehensive robustness report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("No results to report. Run robustness testing first.")
            return
        
        # Generate robustness heatmap
        self._plot_robustness_heatmap(output_path)
        
        # Generate performance degradation curves
        self._plot_degradation_curves(output_path)
        
        # Generate failure mode analysis
        self._plot_failure_analysis(output_path)
        
        # Generate summary report
        self._generate_summary_report(output_path)
        
        print(f"Robustness report generated: {output_path}")
    
    def _plot_robustness_heatmap(self, output_path: Path):
        """Generate robustness heatmap."""
        # Group results by condition type
        condition_groups = {}
        for result in self.results:
            condition_type = result.condition_name.split('_')[0]
            if condition_type not in condition_groups:
                condition_groups[condition_type] = []
            condition_groups[condition_type].append(result)
        
        # Create heatmap for each condition type
        for condition_type, results in condition_groups.items():
            if len(results) < 2:
                continue
            
            # Extract data for heatmap
            condition_names = [r.condition_name for r in results]
            success_rates = [r.success_rate for r in results]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(condition_names)), success_rates)
            plt.xlabel("Condition")
            plt.ylabel("Success Rate")
            plt.title(f"Robustness Heatmap: {condition_type.title()}")
            plt.xticks(range(len(condition_names)), condition_names, rotation=45)
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(success_rates):
                plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / f"robustness_heatmap_{condition_type}.png", dpi=300)
            plt.close()
    
    def _plot_degradation_curves(self, output_path: Path):
        """Generate performance degradation curves."""
        # Group results by condition type and plot degradation
        condition_groups = {}
        for result in self.results:
            condition_type = result.condition_name.split('_')[0]
            if condition_type not in condition_groups:
                condition_groups[condition_type] = []
            condition_groups[condition_type].append(result)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (condition_type, results) in enumerate(condition_groups.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            
            # Sort results by condition parameter
            if condition_type == "control":
                # Sort by frequency
                results.sort(key=lambda x: float(x.condition_name.split('_')[-1].replace('Hz', '')))
                x_values = [float(r.condition_name.split('_')[-1].replace('Hz', '')) for r in results]
                x_label = "Control Frequency (Hz)"
            elif condition_type == "obs":
                # Sort by noise level
                results.sort(key=lambda x: x.condition_params.get('obs_noise', 0))
                x_values = [r.condition_params.get('obs_noise', 0) * 100 for r in results]
                x_label = "Observation Noise (%)"
            elif condition_type == "action":
                # Sort by noise level
                results.sort(key=lambda x: x.condition_params.get('action_noise', 0))
                x_values = [r.condition_params.get('action_noise', 0) * 100 for r in results]
                x_label = "Action Noise (%)"
            else:
                # Sort by parameter value
                results.sort(key=lambda x: list(x.condition_params.values())[0])
                x_values = [list(r.condition_params.values())[0] for r in results]
                x_label = "Parameter Value"
            
            success_rates = [r.success_rate for r in results]
            distances = [r.avg_distance for r in results]
            
            # Plot success rate
            ax2 = ax.twinx()
            line1 = ax.plot(x_values, success_rates, 'b-o', label='Success Rate')
            line2 = ax2.plot(x_values, distances, 'r-s', label='Avg Distance')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel("Success Rate", color='b')
            ax2.set_ylabel("Average Distance (m)", color='r')
            ax.set_title(f"Performance Degradation: {condition_type.title()}")
            ax.set_ylim(0, 1)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_degradation_curves.png", dpi=300)
        plt.close()
    
    def _plot_failure_analysis(self, output_path: Path):
        """Generate failure mode analysis."""
        # Collect all failure modes
        all_failures = []
        for result in self.results:
            all_failures.extend(result.failure_modes)
        
        if not all_failures:
            print("No failures detected for failure analysis.")
            return
        
        # Count failure modes
        failure_counts = {}
        for failure in all_failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        
        # Create pie chart
        plt.figure(figsize=(10, 8))
        labels = list(failure_counts.keys())
        sizes = list(failure_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title("Failure Mode Distribution")
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path / "failure_mode_distribution.png", dpi=300)
        plt.close()
    
    def _generate_summary_report(self, output_path: Path):
        """Generate text summary report."""
        report_path = output_path / "robustness_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("ROBUSTNESS TESTING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total conditions tested: {len(self.results)}\n")
            f.write(f"Episodes per condition: {self.config.episodes_per_condition}\n\n")
            
            # Overall statistics
            success_rates = [r.success_rate for r in self.results]
            avg_success_rate = np.mean(success_rates)
            min_success_rate = np.min(success_rates)
            max_success_rate = np.max(success_rates)
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average success rate: {avg_success_rate:.2%}\n")
            f.write(f"Minimum success rate: {min_success_rate:.2%}\n")
            f.write(f"Maximum success rate: {max_success_rate:.2%}\n")
            f.write(f"Success rate std: {np.std(success_rates):.2%}\n\n")
            
            # Condition-specific results
            f.write("CONDITION-SPECIFIC RESULTS\n")
            f.write("-" * 30 + "\n")
            
            condition_groups = {}
            for result in self.results:
                condition_type = result.condition_name.split('_')[0]
                if condition_type not in condition_groups:
                    condition_groups[condition_type] = []
                condition_groups[condition_type].append(result)
            
            for condition_type, results in condition_groups.items():
                f.write(f"\n{condition_type.upper()} CONDITIONS:\n")
                for result in results:
                    f.write(f"  {result.condition_name}: {result.success_rate:.2%} "
                           f"(distance: {result.avg_distance:.2f}m, "
                           f"reward: {result.avg_reward:.2f})\n")
            
            # Failure analysis
            f.write("\nFAILURE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            all_failures = []
            for result in self.results:
                all_failures.extend(result.failure_modes)
            
            if all_failures:
                failure_counts = {}
                for failure in all_failures:
                    failure_counts[failure] = failure_counts.get(failure, 0) + 1
                
                f.write("Failure mode distribution:\n")
                for failure, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {failure}: {count} occurrences\n")
            else:
                f.write("No failures detected.\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    """Main function for robustness testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Robustness Testing for Passive Walker Models")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"],
                      help="Model backend")
    parser.add_argument("--model-type", type=str, default="mlp", 
                      choices=["mlp", "mlp_large", "lstm", "gru", "bilstm"],
                      help="Model type")
    parser.add_argument("--episodes-per-condition", type=int, default=10,
                      help="Number of episodes per condition")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/robustness",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create robustness configuration
    config = RobustnessConfig(episodes_per_condition=args.episodes_per_condition)
    
    # Initialize tester
    tester = RobustnessTester(config)
    
    # Run robustness testing
    results = tester.test_model_robustness(
        model_path=args.model_path,
        backend=args.backend,
        model_type=args.model_type
    )
    
    # Generate report
    tester.generate_robustness_report(args.output_dir)
    
    print("Robustness testing completed!")


if __name__ == "__main__":
    main()
