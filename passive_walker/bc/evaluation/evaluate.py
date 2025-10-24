"""
Comprehensive Evaluation Suite for BC Models

Enhanced evaluation with detailed metrics, robustness testing, and comparison tools.
Integrates all Phase 3 evaluation components for comprehensive model analysis.
"""

from __future__ import annotations
import numpy as np
import json
import time
import os
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.reward import compute_reward
from passive_walker.bc.config import EvaluationConfig
from passive_walker.bc.utils import set_seed, Normalizer

# Import Phase 3 evaluation tools
try:
    from tools.evaluation.robustness_testing import RobustnessTester, RobustnessConfig
    from tools.evaluation.distribution_shift import DistributionShiftTester, DistributionShiftConfig
    from tools.evaluation.failure_analysis import FailureAnalyzer, FailureAnalysisConfig
    from tools.evaluation.statistical_testing import StatisticalTester, StatisticalTestConfig
    from tools.evaluation.advanced_viz import AdvancedVisualizer, VisualizationConfig
    PHASE3_AVAILABLE = True
except ImportError:
    PHASE3_AVAILABLE = False
    print("Warning: Phase 3 evaluation tools not available. Using basic evaluation only.")


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_id: int
    duration: float
    steps: int
    success: bool
    distance: float
    gait_cycles: int
    avg_reward: float
    total_reward: float
    energy_efficiency: float
    fsm_imitation_error: float
    foot_clearance_avg: float
    velocity_tracking_error: float
    symmetry_error: float
    reward_components: Dict[str, float] = field(default_factory=dict)
    trajectory_data: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Comprehensive evaluation results."""
    model_path: str
    config: Dict[str, Any]
    episodes: List[EpisodeMetrics]
    summary_stats: Dict[str, float]
    robustness_matrix: Dict[str, Dict[str, float]]
    comparison_with_fsm: Dict[str, float]
    timestamp: str
    
    # Phase 3 evaluation results
    robustness_results: Optional[Dict] = None
    distribution_shift_results: Optional[Dict] = None
    failure_analysis_results: Optional[Dict] = None
    statistical_test_results: Optional[Dict] = None
    visualization_data: Optional[Dict] = None


class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for BC models."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = []
        
        # Initialize Phase 3 evaluation tools if available
        if PHASE3_AVAILABLE:
            self.robustness_tester = RobustnessTester()
            self.distribution_tester = DistributionShiftTester()
            self.failure_analyzer = FailureAnalyzer()
            self.statistical_tester = StatisticalTester()
            self.visualizer = AdvancedVisualizer()
        else:
            self.robustness_tester = None
            self.distribution_tester = None
            self.failure_analyzer = None
            self.statistical_tester = None
            self.visualizer = None
        
    def evaluate_model(self, model_path: str, backend: str = "torch") -> EvaluationResults:
        """Evaluate a BC model comprehensively."""
        print(f"Evaluating model: {model_path}")
        
        # Load model and metadata
        model, metadata = self._load_model(model_path, backend)
        
        # Run evaluation episodes
        episodes = []
        for episode_id in range(self.config.episodes):
            print(f"Episode {episode_id + 1}/{self.config.episodes}")
            
            # Test different physics conditions
            for condition in self.config.physics_conditions:
                episode_metrics = self._run_episode(
                    model, metadata, episode_id, condition, backend
                )
                episodes.append(episode_metrics)
        
        # Compute summary statistics
        summary_stats = self._compute_summary_stats(episodes)
        
        # Compute robustness matrix
        robustness_matrix = self._compute_robustness_matrix(episodes)
        
        # Compare with FSM baseline
        fsm_comparison = self._compare_with_fsm()
        
        # Phase 3 evaluations
        robustness_results = None
        distribution_shift_results = None
        failure_analysis_results = None
        statistical_test_results = None
        visualization_data = None
        
        if PHASE3_AVAILABLE:
            print("\n=== Running Phase 3 Evaluations ===")
            
            # Robustness testing
            print("Running robustness testing...")
            robustness_results = self._run_robustness_testing(model_path, backend)
            
            # Distribution shift testing
            print("Running distribution shift analysis...")
            distribution_shift_results = self._run_distribution_shift_testing(model_path, backend)
            
            # Failure analysis
            print("Running failure analysis...")
            failure_analysis_results = self._run_failure_analysis(episodes)
            
            # Statistical testing (if comparing with another model)
            if hasattr(self.config, 'comparison_model') and self.config.comparison_model:
                print("Running statistical testing...")
                statistical_test_results = self._run_statistical_testing(model_path, self.config.comparison_model, backend)
            
            # Prepare visualization data
            print("Preparing visualization data...")
            visualization_data = self._prepare_visualization_data(episodes, robustness_results, distribution_shift_results, failure_analysis_results)
        
        # Create results
        results = EvaluationResults(
            model_path=model_path,
            config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            episodes=episodes,
            summary_stats=summary_stats,
            robustness_matrix=robustness_matrix,
            comparison_with_fsm=fsm_comparison,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            robustness_results=robustness_results,
            distribution_shift_results=distribution_shift_results,
            failure_analysis_results=failure_analysis_results,
            statistical_test_results=statistical_test_results,
            visualization_data=visualization_data
        )
        
        # Save results
        self._save_results(results)
        
        # Generate comprehensive report
        if PHASE3_AVAILABLE and self.visualizer:
            self._generate_comprehensive_report(results)
        
        return results
    
    def _load_model(self, model_path: str, backend: str):
        """Load model and metadata."""
        if backend == "torch":
            return self._load_torch_model(model_path)
        elif backend == "jax":
            return self._load_jax_model(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _load_torch_model(self, model_path: str):
        """Load PyTorch model."""
        import torch
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load metadata
        meta_path = model_path.replace('.pt', '_meta.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct model
        from passive_walker.bc.models.models_torch import create_model
        
        model = create_model(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_sizes=metadata['hidden_sizes'],
            activation=metadata['activation'],
            dropout=metadata.get('dropout', 0.0)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, metadata
    
    def _load_jax_model(self, model_path: str):
        """Load JAX model."""
        import jax
        import equinox as eqx
        
        # Load metadata
        meta_path = model_path.replace('.eqx', '_meta.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Reconstruct model
        from passive_walker.bc.models.models_jax import create_model
        
        model = create_model(
            input_dim=metadata['input_dim'],
            output_dim=metadata['output_dim'],
            hidden_sizes=metadata['hidden_sizes'],
            activation=metadata['activation'],
            dropout=metadata.get('dropout', 0.0)
        )
        
        # Load parameters
        model = eqx.tree_deserialise_leaves(model_path, model)
        
        return model, metadata
    
    def _run_episode(self, model, metadata: Dict, episode_id: int, 
                    physics_condition: str, backend: str) -> EpisodeMetrics:
        """Run a single evaluation episode."""
        # Create environment
        env = PassiveWalkerEnv(
            mode='research' if self.config.use_enhanced_rewards else 'fsm',
            ctrl_hz=self.config.ctrl_hz,
            randomization_profile=self.config.randomization_profile
        )
        
        # Set physics condition
        if physics_condition != "nominal":
            from passive_walker.fsm.collect import PHYSICS_PRESETS
            if physics_condition in PHYSICS_PRESETS:
                physics = PHYSICS_PRESETS[physics_condition]
                env.ramp_deg = physics['ramp_deg']
                env.friction = physics['friction']
                env.randomize_physics = physics['randomize']
        
        # Reset environment
        obs, _ = env.reset(seed=self.config.seed + episode_id)
        
        # Initialize tracking
        trajectory_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'reward_components': [],
            'joint_positions': [],
            'joint_velocities': [],
            'foot_positions': [],
            'fsm_actions': []
        }
        
        total_reward = 0.0
        step_count = 0
        start_time = time.time()
        
        # Run episode
        while env.data.time < self.config.duration_sec:
            # Get model prediction
            if backend == "torch":
                action = self._get_torch_action(model, obs, metadata)
            else:
                action = self._get_jax_action(model, obs, metadata)
            
            # Get FSM action for comparison
            fsm_action = self._get_fsm_action(env)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Track data
            trajectory_data['observations'].append(obs.copy())
            trajectory_data['actions'].append(action.copy())
            trajectory_data['rewards'].append(reward)
            trajectory_data['reward_components'].append(info.copy())
            trajectory_data['joint_positions'].append([
                env.data.qpos[env.qpos_hip],
                env.data.qpos[env.qpos_lk],
                env.data.qpos[env.qpos_rk]
            ])
            trajectory_data['joint_velocities'].append([
                env.data.qvel[env.qvel_hip],
                env.data.qvel[env.qvel_lk],
                env.data.qvel[env.qvel_rk]
            ])
            trajectory_data['foot_positions'].append([
                env.data.xpos[env.b_lfoot, 2],  # Left foot z
                env.data.xpos[env.b_rfoot, 2]   # Right foot z
            ])
            trajectory_data['fsm_actions'].append(fsm_action.copy())
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        duration = time.time() - start_time
        
        # Compute episode metrics
        metrics = self._compute_episode_metrics(
            episode_id, physics_condition, duration, step_count, 
            total_reward, trajectory_data, env
        )
        
        env.close()
        return metrics
    
    def _get_torch_action(self, model, obs: np.ndarray, metadata: Dict) -> np.ndarray:
        """Get action from PyTorch model."""
        import torch
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_tensor = model(obs_tensor)
            action = action_tensor.squeeze(0).numpy()
        
        # Assemble action based on section
        section = metadata.get('section', 'both')
        return self._assemble_action(section, action)
    
    def _get_jax_action(self, model, obs: np.ndarray, metadata: Dict) -> np.ndarray:
        """Get action from JAX model."""
        import jax
        
        action = model(jax.numpy.array(obs))
        action = np.array(action)
        
        # Assemble action based on section
        section = metadata.get('section', 'both')
        return self._assemble_action(section, action)
    
    def _assemble_action(self, section: str, model_out: np.ndarray) -> np.ndarray:
        """Assemble action vector based on control section."""
        if section == "hip":
            return np.array([float(model_out[0]), 0.0, 0.0], dtype=np.float32)
        elif section == "knees":
            return np.array([0.0, float(model_out[0]), float(model_out[1])], dtype=np.float32)
        elif section == "both":
            return np.array([float(model_out[0]), float(model_out[1]), float(model_out[2])], dtype=np.float32)
        else:
            raise ValueError(f"Unknown section: {section}")
    
    def _get_fsm_action(self, env) -> np.ndarray:
        """Get FSM action for comparison."""
        # This is a simplified version - in practice you'd need to implement
        # the FSM logic or use the environment's FSM mode
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def _compute_episode_metrics(self, episode_id: int, physics_condition: str,
                                duration: float, steps: int, total_reward: float,
                                trajectory_data: Dict, env) -> EpisodeMetrics:
        """Compute comprehensive metrics for an episode."""
        
        # Basic metrics
        success = not env.data.time < self.config.duration_sec * 0.8  # Success if >80% duration
        distance = float(env.data.qpos[env.qpos_x])
        
        # Gait cycle detection (simplified)
        gait_cycles = self._count_gait_cycles(trajectory_data['joint_positions'])
        
        # Energy efficiency
        actions = np.array(trajectory_data['actions'])
        energy_efficiency = distance / (np.sum(np.abs(actions)) + 1e-6)
        
        # FSM imitation error
        fsm_actions = np.array(trajectory_data['fsm_actions'])
        bc_actions = np.array(trajectory_data['actions'])
        fsm_imitation_error = np.mean(np.abs(bc_actions - fsm_actions))
        
        # Enhanced reward metrics
        reward_components = {}
        if trajectory_data['reward_components']:
            # Average reward components over episode
            for key in trajectory_data['reward_components'][0].keys():
                if key.startswith('r_'):
                    values = [info[key] for info in trajectory_data['reward_components']]
                    reward_components[key] = np.mean(values)
        
        # Foot clearance metrics
        foot_positions = np.array(trajectory_data['foot_positions'])
        foot_clearance_avg = np.mean(np.maximum(foot_positions[:, 0], foot_positions[:, 1]))
        
        # Velocity tracking error
        velocity_tracking_error = 0.0
        if 'r_velocity' in reward_components:
            velocity_tracking_error = 1.0 - reward_components['r_velocity']  # Convert to error
        
        # Symmetry error
        symmetry_error = 0.0
        if 'r_symmetry' in reward_components:
            symmetry_error = 1.0 - reward_components['r_symmetry']  # Convert to error
        
        return EpisodeMetrics(
            episode_id=episode_id,
            duration=duration,
            steps=steps,
            success=success,
            distance=distance,
            gait_cycles=gait_cycles,
            avg_reward=total_reward / max(steps, 1),
            total_reward=total_reward,
            energy_efficiency=energy_efficiency,
            fsm_imitation_error=fsm_imitation_error,
            foot_clearance_avg=foot_clearance_avg,
            velocity_tracking_error=velocity_tracking_error,
            symmetry_error=symmetry_error,
            reward_components=reward_components,
            trajectory_data=trajectory_data
        )
    
    def _count_gait_cycles(self, joint_positions: List[List[float]]) -> int:
        """Count gait cycles from joint position data."""
        if len(joint_positions) < 10:
            return 0
        
        # Simple gait cycle detection based on hip angle oscillations
        hip_angles = [pos[0] for pos in joint_positions]
        
        # Find peaks and valleys
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hip_angles)
        valleys, _ = find_peaks([-x for x in hip_angles])
        
        # Count cycles (each peak-valley pair is half a cycle)
        return min(len(peaks), len(valleys))
    
    def _compute_summary_stats(self, episodes: List[EpisodeMetrics]) -> Dict[str, float]:
        """Compute summary statistics across all episodes."""
        if not episodes:
            return {}
        
        return {
            'success_rate': np.mean([ep.success for ep in episodes]),
            'avg_distance': np.mean([ep.distance for ep in episodes]),
            'avg_duration': np.mean([ep.duration for ep in episodes]),
            'avg_gait_cycles': np.mean([ep.gait_cycles for ep in episodes]),
            'avg_reward': np.mean([ep.avg_reward for ep in episodes]),
            'avg_energy_efficiency': np.mean([ep.energy_efficiency for ep in episodes]),
            'avg_fsm_imitation_error': np.mean([ep.fsm_imitation_error for ep in episodes]),
            'avg_foot_clearance': np.mean([ep.foot_clearance_avg for ep in episodes]),
            'avg_velocity_tracking_error': np.mean([ep.velocity_tracking_error for ep in episodes]),
            'avg_symmetry_error': np.mean([ep.symmetry_error for ep in episodes]),
        }
    
    def _compute_robustness_matrix(self, episodes: List[EpisodeMetrics]) -> Dict[str, Dict[str, float]]:
        """Compute robustness matrix across physics conditions."""
        # Group episodes by physics condition
        condition_groups = {}
        for ep in episodes:
            condition = getattr(ep, 'physics_condition', 'nominal')
            if condition not in condition_groups:
                condition_groups[condition] = []
            condition_groups[condition].append(ep)
        
        # Compute metrics per condition
        robustness_matrix = {}
        for condition, eps in condition_groups.items():
            robustness_matrix[condition] = {
                'success_rate': np.mean([ep.success for ep in eps]),
                'avg_distance': np.mean([ep.distance for ep in eps]),
                'avg_reward': np.mean([ep.avg_reward for ep in eps]),
                'episode_count': len(eps)
            }
        
        return robustness_matrix
    
    def _compare_with_fsm(self) -> Dict[str, float]:
        """Compare BC performance with FSM baseline."""
        # This would run FSM evaluation and compare
        # For now, return placeholder values
        return {
            'fsm_success_rate': 0.95,
            'fsm_avg_distance': 15.0,
            'fsm_avg_reward': 2.5,
            'bc_vs_fsm_success': 0.0,  # Would be computed
            'bc_vs_fsm_distance': 0.0,  # Would be computed
            'bc_vs_fsm_reward': 0.0,   # Would be computed
        }
    
    def _save_results(self, results: EvaluationResults):
        """Save evaluation results."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Save detailed results
        results_path = os.path.join(self.config.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert to serializable format
            results_dict = {
                'model_path': results.model_path,
                'config': results.config,
                'summary_stats': results.summary_stats,
                'robustness_matrix': results.robustness_matrix,
                'comparison_with_fsm': results.comparison_with_fsm,
                'timestamp': results.timestamp,
                'episode_count': len(results.episodes)
            }
            json.dump(results_dict, f, indent=2)
        
        # Save trajectory data if requested
        if self.config.save_trajectories:
            for i, episode in enumerate(results.episodes):
                traj_path = os.path.join(self.config.output_dir, f'episode_{i:03d}.npz')
                np.savez(traj_path, **episode.trajectory_data)
        
        print(f"Results saved to: {self.config.output_dir}")
    
    def _run_robustness_testing(self, model_path: str, backend: str) -> Dict:
        """Run robustness testing using Phase 3 tools."""
        if not self.robustness_tester:
            return None
        
        try:
            # Determine model type from path
            model_type = "mlp"  # Default
            if "lstm" in model_path.lower():
                model_type = "lstm"
            elif "gru" in model_path.lower():
                model_type = "gru"
            elif "bilstm" in model_path.lower():
                model_type = "bilstm"
            
            # Run robustness testing
            results = self.robustness_tester.test_model_robustness(
                model_path=model_path,
                backend=backend,
                model_type=model_type
            )
            
            # Generate robustness report
            robustness_output_dir = os.path.join(self.config.output_dir, "robustness")
            self.robustness_tester.generate_robustness_report(robustness_output_dir)
            
            return {
                "results": results,
                "output_dir": robustness_output_dir,
                "summary": {
                    "total_conditions": len(results),
                    "avg_success_rate": np.mean([r.success_rate for r in results.values()]),
                    "min_success_rate": np.min([r.success_rate for r in results.values()]),
                    "max_success_rate": np.max([r.success_rate for r in results.values()])
                }
            }
        except Exception as e:
            print(f"Robustness testing failed: {e}")
            return None
    
    def _run_distribution_shift_testing(self, model_path: str, backend: str) -> Dict:
        """Run distribution shift testing using Phase 3 tools."""
        if not self.distribution_tester:
            return None
        
        try:
            # Determine model type from path
            model_type = "mlp"  # Default
            if "lstm" in model_path.lower():
                model_type = "lstm"
            elif "gru" in model_path.lower():
                model_type = "gru"
            
            # Run distribution shift testing
            results = self.distribution_tester.test_distribution_shift(
                model_path=model_path,
                backend=backend,
                model_type=model_type
            )
            
            # Generate distribution shift report
            distribution_output_dir = os.path.join(self.config.output_dir, "distribution_shift")
            self.distribution_tester.generate_distribution_report(distribution_output_dir)
            
            return {
                "results": results,
                "output_dir": distribution_output_dir,
                "summary": {
                    "total_conditions": len(results),
                    "avg_kl_divergence": np.mean([r.kl_divergence for r in results.values()]),
                    "avg_success_rate": np.mean([r.success_rate for r in results.values()])
                }
            }
        except Exception as e:
            print(f"Distribution shift testing failed: {e}")
            return None
    
    def _run_failure_analysis(self, episodes: List[EpisodeMetrics]) -> Dict:
        """Run failure analysis using Phase 3 tools."""
        if not self.failure_analyzer:
            return None
        
        try:
            # Convert episodes to format expected by failure analyzer
            episode_data = []
            for episode in episodes:
                episode_dict = {
                    "observations": episode.trajectory_data.get("observations", []),
                    "actions": episode.trajectory_data.get("actions", []),
                    "rewards": episode.trajectory_data.get("rewards", []),
                    "done": not episode.success
                }
                episode_data.append(episode_dict)
            
            # Run failure analysis
            result = self.failure_analyzer.analyze_failures(episode_data)
            
            # Generate failure analysis report
            failure_output_dir = os.path.join(self.config.output_dir, "failure_analysis")
            self.failure_analyzer.generate_failure_report(result, failure_output_dir)
            
            return {
                "result": result,
                "output_dir": failure_output_dir,
                "summary": {
                    "total_episodes": result.total_episodes,
                    "total_failures": result.total_failures,
                    "failure_rate": result.failure_rate,
                    "failure_distribution": result.failure_distribution
                }
            }
        except Exception as e:
            print(f"Failure analysis failed: {e}")
            return None
    
    def _run_statistical_testing(self, model_a_path: str, model_b_path: str, backend: str) -> Dict:
        """Run statistical testing between two models."""
        if not self.statistical_tester:
            return None
        
        try:
            # This would require running evaluation on both models
            # For now, return a placeholder
            return {
                "model_a": model_a_path,
                "model_b": model_b_path,
                "status": "placeholder - requires dual model evaluation"
            }
        except Exception as e:
            print(f"Statistical testing failed: {e}")
            return None
    
    def _prepare_visualization_data(self, episodes: List[EpisodeMetrics], 
                                  robustness_results: Dict, distribution_shift_results: Dict,
                                  failure_analysis_results: Dict) -> Dict:
        """Prepare comprehensive data for visualization."""
        try:
            # Extract trajectory data
            trajectory_data = None
            if episodes and episodes[0].trajectory_data:
                first_episode = episodes[0]
                trajectory_data = {
                    'x': [obs[0] for obs in first_episode.trajectory_data.get('observations', [])],
                    'z': [obs[1] for obs in first_episode.trajectory_data.get('observations', [])],  # Z position
                    'pitch': [obs[2] for obs in first_episode.trajectory_data.get('observations', [])]  # Pitch angle
                }
            
            # Extract contact data
            contact_data = None
            if episodes and episodes[0].trajectory_data:
                first_episode = episodes[0]
                contact_data = {
                    'left_contact': [obs[11] for obs in first_episode.trajectory_data.get('observations', [])],
                    'right_contact': [obs[12] for obs in first_episode.trajectory_data.get('observations', [])],
                    'left_force': [obs[13] for obs in first_episode.trajectory_data.get('observations', [])],
                    'right_force': [obs[14] for obs in first_episode.trajectory_data.get('observations', [])]
                }
            
            # Extract gait data
            gait_data = {
                'step_lengths': [ep.distance / max(ep.gait_cycles, 1) for ep in episodes],
                'stance_durations': [ep.duration / max(ep.gait_cycles, 1) for ep in episodes],
                'swing_durations': [ep.duration / max(ep.gait_cycles, 1) * 0.4 for ep in episodes]
            }
            
            # Extract model comparison data
            model_comparison = {
                'Current Model': {
                    'success_rate': [ep.success for ep in episodes],
                    'distance': [ep.distance for ep in episodes],
                    'reward': [ep.total_reward for ep in episodes]
                }
            }
            
            # Extract robustness data
            robustness_data = None
            if robustness_results and robustness_results.get("results"):
                robustness_data = {}
                for condition, result in robustness_results["results"].items():
                    robustness_data[condition] = {
                        'success_rate': result.success_rate,
                        'distance': result.avg_distance,
                        'reward': result.avg_reward
                    }
            
            # Extract failure distribution
            failure_distribution = None
            if failure_analysis_results and failure_analysis_results.get("result"):
                failure_distribution = failure_analysis_results["result"].failure_distribution
            
            return {
                'trajectory_data': trajectory_data,
                'contact_data': contact_data,
                'gait_data': gait_data,
                'model_comparison': model_comparison,
                'robustness_data': robustness_data,
                'failure_distribution': failure_distribution
            }
        except Exception as e:
            print(f"Visualization data preparation failed: {e}")
            return None
    
    def _generate_comprehensive_report(self, results: EvaluationResults):
        """Generate comprehensive visualization report."""
        if not self.visualizer or not results.visualization_data:
            return
        
        try:
            visualization_output_dir = os.path.join(self.config.output_dir, "visualization")
            self.visualizer.generate_comprehensive_report(
                results.visualization_data, 
                visualization_output_dir
            )
            print(f"Comprehensive visualization report generated: {visualization_output_dir}")
        except Exception as e:
            print(f"Comprehensive report generation failed: {e}")


def evaluate_model_comprehensive(checkpoint_path: str, config: EvaluationConfig) -> EvaluationResults:
    """Main function for comprehensive model evaluation."""
    evaluator = ComprehensiveEvaluator(config)
    return evaluator.evaluate_model(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive BC Model Evaluation with Phase 3 Tools")
    
    # Basic evaluation arguments
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "jax"],
                      help="Model backend")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/evaluation",
                      help="Output directory for results")
    
    # Evaluation configuration
    parser.add_argument("--episodes", type=int, default=10,
                      help="Number of evaluation episodes")
    parser.add_argument("--duration-sec", type=float, default=25.0,
                      help="Episode duration in seconds")
    parser.add_argument("--ctrl-hz", type=int, default=100,
                      help="Control frequency")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    # Phase 3 evaluation options
    parser.add_argument("--enable-phase3", action="store_true", default=True,
                      help="Enable Phase 3 comprehensive evaluation")
    parser.add_argument("--enable-robustness", action="store_true", default=True,
                      help="Enable robustness testing")
    parser.add_argument("--enable-distribution-shift", action="store_true", default=True,
                      help="Enable distribution shift analysis")
    parser.add_argument("--enable-failure-analysis", action="store_true", default=True,
                      help="Enable failure analysis")
    parser.add_argument("--enable-visualization", action="store_true", default=True,
                      help="Enable advanced visualization")
    
    # Comparison options
    parser.add_argument("--comparison-model", type=str, default=None,
                      help="Path to comparison model for statistical testing")
    
    # Physics conditions
    parser.add_argument("--physics-conditions", nargs="+", 
                      default=["nominal", "gentle", "steep"],
                      help="Physics conditions to test")
    
    # Advanced options
    parser.add_argument("--save-trajectories", action="store_true",
                      help="Save detailed trajectory data")
    parser.add_argument("--use-enhanced-rewards", action="store_true", default=True,
                      help="Use enhanced reward function")
    
    args = parser.parse_args()
    
    # Create evaluation configuration
    config = EvaluationConfig(
        checkpoint_path=args.model_path,
        episodes=args.episodes,
        duration_sec=args.duration_sec,
        ctrl_hz=args.ctrl_hz,
        seed=args.seed,
        output_dir=args.output_dir,
        physics_conditions=args.physics_conditions,
        save_trajectories=args.save_trajectories,
        use_enhanced_rewards=args.use_enhanced_rewards
    )
    
    # Add comparison model if provided
    if args.comparison_model:
        config.comparison_model = args.comparison_model
    
    # Disable Phase 3 components if requested
    if not args.enable_phase3:
        # Note: This would require restructuring to properly disable Phase 3
        # For now, we'll just print a warning
        print("Warning: Phase 3 evaluation disabled via CLI argument")
    
    print("="*80)
    print("COMPREHENSIVE BC MODEL EVALUATION")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Backend: {args.backend}")
    print(f"Episodes: {args.episodes}")
    print(f"Duration: {args.duration_sec}s")
    print(f"Phase 3 Evaluation: {'Enabled' if PHASE3_AVAILABLE else 'Disabled'}")
    print("="*80)
    
    # Run comprehensive evaluation
    try:
        results = evaluate_model_comprehensive(args.model_path, config)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"Success Rate: {results.summary_stats['success_rate']:.1%}")
        print(f"Average Distance: {results.summary_stats['avg_distance']:.2f}m")
        print(f"Average Reward: {results.summary_stats['avg_reward']:.3f}")
        print(f"Energy Efficiency: {results.summary_stats['avg_energy_efficiency']:.3f}")
        
        if PHASE3_AVAILABLE and results.robustness_results:
            print(f"\nRobustness Testing:")
            print(f"  Conditions Tested: {results.robustness_results['summary']['total_conditions']}")
            print(f"  Average Success Rate: {results.robustness_results['summary']['avg_success_rate']:.1%}")
            print(f"  Success Rate Range: {results.robustness_results['summary']['min_success_rate']:.1%} - {results.robustness_results['summary']['max_success_rate']:.1%}")
        
        if PHASE3_AVAILABLE and results.failure_analysis_results:
            print(f"\nFailure Analysis:")
            print(f"  Total Failures: {results.failure_analysis_results['summary']['total_failures']}")
            print(f"  Failure Rate: {results.failure_analysis_results['summary']['failure_rate']:.1%}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

