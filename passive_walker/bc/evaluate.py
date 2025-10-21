"""
Comprehensive Evaluation Suite for BC Models

Enhanced evaluation with detailed metrics, robustness testing, and comparison tools.
"""

from __future__ import annotations
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.reward import compute_reward
from passive_walker.bc.config import EvaluationConfig
from passive_walker.bc.utils import set_seed, Normalizer


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


class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for BC models."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = []
        
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
        
        # Create results
        results = EvaluationResults(
            model_path=model_path,
            config=self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            episodes=episodes,
            summary_stats=summary_stats,
            robustness_matrix=robustness_matrix,
            comparison_with_fsm=fsm_comparison,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Save results
        self._save_results(results)
        
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


def evaluate_model_comprehensive(checkpoint_path: str, config: EvaluationConfig) -> EvaluationResults:
    """Main function for comprehensive model evaluation."""
    evaluator = ComprehensiveEvaluator(config)
    return evaluator.evaluate_model(checkpoint_path)


if __name__ == "__main__":
    # Example usage
    config = EvaluationConfig(
        checkpoint_path="experiments/models/torch_both_seed123_ep1_steps180000.pt",
        episodes=5,
        duration_sec=25.0,
        physics_conditions=["nominal", "gentle", "steep"],
        use_enhanced_rewards=True
    )
    
    results = evaluate_model_comprehensive(config.checkpoint_path, config)
    print(f"Evaluation complete. Success rate: {results.summary_stats['success_rate']:.1%}")

