#!/usr/bin/env python3
"""
Distribution Shift Testing for Passive Walker Models

This module implements cross-condition generalization analysis:
- Train/test distribution analysis
- Cross-condition evaluation matrix
- Distribution shift detection using KL divergence
- Out-of-distribution state identification
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import stats
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.physics_conditions import PhysicsConditionManager, PhysicsParameter


@dataclass
class DistributionShiftConfig:
    """Configuration for distribution shift testing."""
    # Training conditions
    train_conditions: List[Dict[str, float]] = None
    
    # Test conditions
    test_conditions: List[Dict[str, float]] = None
    
    # Evaluation parameters
    episodes_per_condition: int = 20
    max_steps_per_episode: int = 1000
    dt: float = 0.01
    
    # Distribution analysis parameters
    kl_divergence_bins: int = 50
    tsne_perplexity: float = 30.0
    tsne_n_components: int = 2
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.train_conditions is None:
            self.train_conditions = [
                {"gravity": 1.0, "mass": 1.0, "friction": 0.7, "damping": 0.5}  # Nominal
            ]
        
        if self.test_conditions is None:
            self.test_conditions = [
                {"gravity": 1.0, "mass": 1.0, "friction": 0.7, "damping": 0.5},  # Nominal
                {"gravity": 0.8, "mass": 1.0, "friction": 0.7, "damping": 0.5},  # Low gravity
                {"gravity": 1.2, "mass": 1.0, "friction": 0.7, "damping": 0.5},  # High gravity
                {"gravity": 1.0, "mass": 0.8, "friction": 0.7, "damping": 0.5},  # Low mass
                {"gravity": 1.0, "mass": 1.2, "friction": 0.7, "damping": 0.5},  # High mass
                {"gravity": 1.0, "mass": 1.0, "friction": 0.5, "damping": 0.5},  # Low friction
                {"gravity": 1.0, "mass": 1.0, "friction": 0.9, "damping": 0.5},  # High friction
            ]


@dataclass
class DistributionShiftResult:
    """Results from distribution shift analysis."""
    condition_name: str
    condition_params: Dict[str, float]
    metrics: Dict[str, float]
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success_rate: float
    avg_distance: float
    avg_reward: float
    kl_divergence: float
    distribution_stats: Dict[str, float]


class DistributionShiftTester:
    """Cross-condition generalization analysis for passive walker models."""
    
    def __init__(self, config: Optional[DistributionShiftConfig] = None):
        """Initialize distribution shift tester.
        
        Args:
            config: Distribution shift testing configuration
        """
        self.config = config or DistributionShiftConfig()
        self.results: List[DistributionShiftResult] = []
        self.physics_manager = PhysicsConditionManager()
        self.train_observations: Optional[np.ndarray] = None
        
    def test_distribution_shift(self, 
                              model_path: str,
                              backend: str = "torch",
                              model_type: str = "mlp") -> Dict[str, DistributionShiftResult]:
        """Test model generalization across different conditions.
        
        Args:
            model_path: Path to model checkpoint
            backend: "torch" or "jax"
            model_type: "mlp", "lstm", "gru", "bilstm"
            
        Returns:
            Dictionary mapping condition names to results
        """
        print(f"Starting distribution shift testing for {model_type} model ({backend})")
        print(f"Model path: {model_path}")
        
        # Load model
        model = self._load_model(model_path, backend, model_type)
        
        # Collect training data distribution (if not already collected)
        if self.train_observations is None:
            print("\n=== Collecting Training Distribution ===")
            self.train_observations = self._collect_observations(
                model, backend, self.config.train_conditions[0], "train"
            )
        
        # Test on all conditions
        all_results = {}
        
        print("\n=== Testing Cross-Condition Generalization ===")
        for condition in self.config.test_conditions:
            condition_name = self._get_condition_name(condition)
            print(f"Testing condition: {condition_name}")
            
            # Collect observations for this condition
            observations = self._collect_observations(model, backend, condition, condition_name)
            
            # Run evaluation episodes
            episodes = self._run_episodes(model, backend, condition, condition_name)
            
            # Compute metrics
            metrics = self._compute_metrics(episodes)
            
            # Compute distribution shift
            kl_divergence = self._compute_kl_divergence(observations)
            
            # Compute distribution statistics
            dist_stats = self._compute_distribution_stats(observations)
            
            result = DistributionShiftResult(
                condition_name=condition_name,
                condition_params=condition,
                metrics=metrics,
                observations=observations,
                actions=np.array([action for ep in episodes for action in ep["actions"]]),
                rewards=np.array([reward for ep in episodes for reward in ep["rewards"]]),
                success_rate=metrics["success_rate"],
                avg_distance=metrics["avg_distance"],
                avg_reward=metrics["avg_reward"],
                kl_divergence=kl_divergence,
                distribution_stats=dist_stats
            )
            
            all_results[condition_name] = result
            print(f"  Success rate: {result.success_rate:.2%}, KL divergence: {result.kl_divergence:.4f}")
        
        # Store all results
        self.results.extend(all_results.values())
        
        print(f"\nDistribution shift testing completed: {len(all_results)} conditions tested")
        return all_results
    
    def _load_model(self, model_path: str, backend: str, model_type: str):
        """Load model from checkpoint."""
        if backend == "torch":
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if model_type == "mlp":
                from passive_walker.bc.models.models_torch import TorchMLP
                model = TorchMLP(input_dim=17, hidden_dim=256, output_dim=3)
            elif model_type == "mlp_large":
                from passive_walker.bc.models.models_torch import TorchMLPLarge
                model = TorchMLPLarge(input_dim=17, output_dim=3)
            elif model_type == "lstm":
                from passive_walker.bc.models.temporal_torch import TorchLSTM
                model = TorchLSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "gru":
                from passive_walker.bc.models.temporal_torch import TorchGRU
                model = TorchGRU(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "bilstm":
                from passive_walker.bc.models.temporal_torch import TorchBiLSTM
                model = TorchBiLSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return model
            
        elif backend == "jax":
            import equinox as eqx
            
            if model_type == "mlp":
                from passive_walker.bc.models.models_jax import MLP
                model = MLP(input_dim=17, hidden_dim=256, output_dim=3)
            elif model_type == "lstm":
                from passive_walker.bc.models.temporal_jax import LSTM
                model = LSTM(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            elif model_type == "gru":
                from passive_walker.bc.models.temporal_jax import GRU
                model = GRU(input_dim=17, hidden_dim=128, output_dim=3, num_layers=2)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model = eqx.tree_deserialise_leaves(model_path, model)
            return model
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _collect_observations(self, model, backend: str, condition: Dict[str, float], 
                            condition_name: str) -> np.ndarray:
        """Collect observations for distribution analysis."""
        # Create environment with specific condition
        env = PassiveWalkerEnv()
        self.physics_manager.apply_condition_to_env(env, condition)
        
        observations = []
        
        for episode_idx in range(self.config.episodes_per_condition):
            obs, info = env.reset()
            observations.append(obs.copy())
            
            step_count = 0
            while step_count < self.config.max_steps_per_episode:
                # Get action from model
                action = self._get_model_action(model, obs, backend)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                observations.append(next_obs.copy())
                
                if terminated or truncated:
                    break
                
                obs = next_obs
                step_count += 1
        
        return np.array(observations)
    
    def _run_episodes(self, model, backend: str, condition: Dict[str, float], 
                     condition_name: str) -> List[Dict]:
        """Run evaluation episodes."""
        # Create environment with specific condition
        env = PassiveWalkerEnv()
        self.physics_manager.apply_condition_to_env(env, condition)
        
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
                # Get action from model
                action = self._get_model_action(model, obs, backend)
                
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
    
    def _compute_kl_divergence(self, test_observations: np.ndarray) -> float:
        """Compute KL divergence between train and test distributions."""
        if self.train_observations is None:
            return 0.0
        
        # Compute histograms for each dimension
        kl_divs = []
        
        for dim in range(min(self.train_observations.shape[1], test_observations.shape[1])):
            train_data = self.train_observations[:, dim]
            test_data = test_observations[:, dim]
            
            # Create bins
            min_val = min(np.min(train_data), np.min(test_data))
            max_val = max(np.max(train_data), np.max(test_data))
            bins = np.linspace(min_val, max_val, self.config.kl_divergence_bins)
            
            # Compute histograms
            train_hist, _ = np.histogram(train_data, bins=bins, density=True)
            test_hist, _ = np.histogram(test_data, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            train_hist = train_hist + epsilon
            test_hist = test_hist + epsilon
            
            # Normalize
            train_hist = train_hist / np.sum(train_hist)
            test_hist = test_hist / np.sum(test_hist)
            
            # Compute KL divergence
            kl_div = np.sum(train_hist * np.log(train_hist / test_hist))
            kl_divs.append(kl_div)
        
        return np.mean(kl_divs)
    
    def _compute_distribution_stats(self, observations: np.ndarray) -> Dict[str, float]:
        """Compute distribution statistics."""
        return {
            "mean": np.mean(observations),
            "std": np.std(observations),
            "min": np.min(observations),
            "max": np.max(observations),
            "median": np.median(observations),
            "q25": np.percentile(observations, 25),
            "q75": np.percentile(observations, 75)
        }
    
    def _get_condition_name(self, condition: Dict[str, float]) -> str:
        """Generate condition name from parameters."""
        if len(condition) == 1:
            param, value = list(condition.items())[0]
            return f"{param}_{value:.2f}"
        else:
            # Multiple parameters
            name_parts = []
            for param, value in condition.items():
                if value != 1.0:  # Only include non-default values
                    name_parts.append(f"{param}_{value:.2f}")
            return "_".join(name_parts) if name_parts else "nominal"
    
    def generate_distribution_report(self, output_dir: str = "experiments/outputs/distribution_shift"):
        """Generate comprehensive distribution shift report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not self.results:
            print("No results to report. Run distribution shift testing first.")
            return
        
        # Generate cross-condition performance matrix
        self._plot_cross_condition_matrix(output_path)
        
        # Generate performance vs distribution shift plot
        self._plot_performance_vs_shift(output_path)
        
        # Generate distribution visualization
        self._plot_distribution_visualization(output_path)
        
        # Generate summary report
        self._generate_summary_report(output_path)
        
        print(f"Distribution shift report generated: {output_path}")
    
    def _plot_cross_condition_matrix(self, output_path: Path):
        """Generate cross-condition performance matrix."""
        # Create performance matrix
        condition_names = [r.condition_name for r in self.results]
        success_rates = [r.success_rate for r in self.results]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(condition_names)), success_rates)
        plt.xlabel("Test Condition")
        plt.ylabel("Success Rate")
        plt.title("Cross-Condition Performance Matrix")
        plt.xticks(range(len(condition_names)), condition_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(success_rates):
            plt.text(i, v + 0.01, f"{v:.2%}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / "cross_condition_performance.png", dpi=300)
        plt.close()
    
    def _plot_performance_vs_shift(self, output_path: Path):
        """Generate performance vs distribution shift scatter plot."""
        kl_divs = [r.kl_divergence for r in self.results]
        success_rates = [r.success_rate for r in self.results]
        distances = [r.avg_distance for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate vs KL divergence
        ax1.scatter(kl_divs, success_rates, alpha=0.7, s=100)
        ax1.set_xlabel("KL Divergence")
        ax1.set_ylabel("Success Rate")
        ax1.set_title("Performance vs Distribution Shift")
        ax1.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(kl_divs, success_rates)[0, 1]
        ax1.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Distance vs KL divergence
        ax2.scatter(kl_divs, distances, alpha=0.7, s=100, color='red')
        ax2.set_xlabel("KL Divergence")
        ax2.set_ylabel("Average Distance (m)")
        ax2.set_title("Distance vs Distribution Shift")
        ax2.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = np.corrcoef(kl_divs, distances)[0, 1]
        ax2.text(0.05, 0.95, f"Correlation: {corr_coef:.3f}", 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_vs_distribution_shift.png", dpi=300)
        plt.close()
    
    def _plot_distribution_visualization(self, output_path: Path):
        """Generate distribution visualization using t-SNE."""
        if len(self.results) < 2:
            print("Not enough results for distribution visualization.")
            return
        
        # Combine all observations
        all_observations = []
        all_labels = []
        
        for result in self.results:
            all_observations.append(result.observations)
            all_labels.extend([result.condition_name] * len(result.observations))
        
        # Concatenate observations
        combined_obs = np.vstack(all_observations)
        
        if HAS_SKLEARN:
            # Reduce dimensionality using PCA first (for efficiency)
            pca = PCA(n_components=min(50, combined_obs.shape[1]))
            obs_pca = pca.fit_transform(combined_obs)
            
            # Apply t-SNE
            tsne = TSNE(n_components=self.config.tsne_n_components, 
                       perplexity=self.config.tsne_perplexity, random_state=42)
            obs_tsne = tsne.fit_transform(obs_pca)
        else:
            # Fallback: use first 2 dimensions
            obs_tsne = combined_obs[:, :2]
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 8))
        
        # Get unique labels and colors
        unique_labels = list(set(all_labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(all_labels) == label
            plt.scatter(obs_tsne[mask, 0], obs_tsne[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.6, s=20)
        
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("Observation Distribution Visualization (t-SNE)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "distribution_visualization_tsne.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, output_path: Path):
        """Generate text summary report."""
        report_path = output_path / "distribution_shift_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("DISTRIBUTION SHIFT TESTING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total conditions tested: {len(self.results)}\n")
            f.write(f"Episodes per condition: {self.config.episodes_per_condition}\n\n")
            
            # Overall statistics
            success_rates = [r.success_rate for r in self.results]
            kl_divs = [r.kl_divergence for r in self.results]
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average success rate: {np.mean(success_rates):.2%}\n")
            f.write(f"Success rate std: {np.std(success_rates):.2%}\n")
            f.write(f"Average KL divergence: {np.mean(kl_divs):.4f}\n")
            f.write(f"KL divergence std: {np.std(kl_divs):.4f}\n\n")
            
            # Correlation analysis
            corr_coef = np.corrcoef(kl_divs, success_rates)[0, 1]
            f.write("CORRELATION ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Success rate vs KL divergence correlation: {corr_coef:.3f}\n\n")
            
            # Condition-specific results
            f.write("CONDITION-SPECIFIC RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for result in self.results:
                f.write(f"{result.condition_name}:\n")
                f.write(f"  Success rate: {result.success_rate:.2%}\n")
                f.write(f"  Average distance: {result.avg_distance:.2f}m\n")
                f.write(f"  Average reward: {result.avg_reward:.2f}\n")
                f.write(f"  KL divergence: {result.kl_divergence:.4f}\n")
                f.write(f"  Distribution stats: mean={result.distribution_stats['mean']:.3f}, "
                       f"std={result.distribution_stats['std']:.3f}\n\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    """Main function for distribution shift testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distribution Shift Testing for Passive Walker Models")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"],
                      help="Model backend")
    parser.add_argument("--model-type", type=str, default="mlp", 
                      choices=["mlp", "mlp_large", "lstm", "gru", "bilstm"],
                      help="Model type")
    parser.add_argument("--episodes-per-condition", type=int, default=20,
                      help="Number of episodes per condition")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/distribution_shift",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create distribution shift configuration
    config = DistributionShiftConfig(episodes_per_condition=args.episodes_per_condition)
    
    # Initialize tester
    tester = DistributionShiftTester(config)
    
    # Run distribution shift testing
    results = tester.test_distribution_shift(
        model_path=args.model_path,
        backend=args.backend,
        model_type=args.model_type
    )
    
    # Generate report
    tester.generate_distribution_report(args.output_dir)
    
    print("Distribution shift testing completed!")


if __name__ == "__main__":
    main()
