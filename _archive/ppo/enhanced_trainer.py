"""
Enhanced PPO Training with Environment Integration

Integrates PPO with enhanced environment features: research rewards, 
domain randomization, curriculum learning, and online perturbations.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import time
import os

from .trainer import PPOTrainer
from .config import PPOConfig
from .models import create_actor_critic
from ..core.env import PassiveWalkerEnv
from ..core.environment_enhancements import (
    OnlinePerturbationManager, AdaptiveRandomizationManager, RewardCurriculumManager
)
from ..core.randomization import get_randomization_config
from ..bc.experiment.tracking import ExperimentTracker


class EnhancedPPOTrainer(PPOTrainer):
    """
    Enhanced PPO trainer with environment integration.
    
    Supports research rewards, domain randomization, curriculum learning,
    and online perturbations.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: PPOConfig,
                 device: str = "cpu",
                 tracker: Optional[ExperimentTracker] = None,
                 output_dir: str = "ppo_runs"):
        """
        Initialize enhanced PPO trainer.
        
        Args:
            model: Actor-critic model
            config: PPO configuration
            device: Device to train on
            tracker: Experiment tracker for logging
            output_dir: Directory to save outputs
        """
        super().__init__(model, config, device, tracker, output_dir)
        
        # Environment enhancement components
        self.perturbation_manager = None
        self.randomization_manager = None
        self.reward_curriculum_manager = None
        
        # Training statistics
        self.episode_success_rate = []
        self.episode_lengths = []
        self.environment_stats = {
            "perturbation_applied": 0,
            "randomization_strength": [],
            "curriculum_stage": []
        }
    
    def setup_environment_enhancements(self):
        """Setup environment enhancement components."""
        # Online perturbation manager
        if self.config.use_curriculum:
            self.perturbation_manager = OnlinePerturbationManager(
                impulse_probability=0.1,
                impulse_magnitude=50.0,
                continuous_push_probability=0.05,
                continuous_push_magnitude=10.0,
                terrain_change_probability=0.02
            )
        
        # Adaptive randomization manager
        if self.config.use_domain_randomization:
            self.randomization_manager = AdaptiveRandomizationManager(
                base_randomization_strength=0.5,
                adaptation_rate=0.1,
                min_strength=0.1,
                max_strength=1.0
            )
        
        # Reward curriculum manager
        if self.config.use_curriculum:
            total_episodes = self.config.total_timesteps // self.config.n_steps
            self.reward_curriculum_manager = RewardCurriculumManager(
                total_episodes=total_episodes,
                transition_episodes=total_episodes // 4,
                start_mode="fsm",
                end_mode="research"
            )
    
    def create_enhanced_environment(self) -> PassiveWalkerEnv:
        """
        Create enhanced environment with randomization and curriculum.
        
        Returns:
            Enhanced PassiveWalkerEnv instance
        """
        # Base environment configuration
        env_kwargs = self.config.env_kwargs.copy()
        
        # Set environment mode based on curriculum
        # If curriculum is enabled, start with FSM and transition to research
        # If no curriculum, use research mode directly for better rewards
        if "mode" not in env_kwargs:
            if self.config.use_curriculum:
                env_kwargs["mode"] = "fsm"  # Will transition to research via curriculum
            else:
                env_kwargs["mode"] = "research"  # Use research rewards directly
        
        # Add domain randomization
        if self.config.use_domain_randomization:
            randomization_config = get_randomization_config(self.config.randomization_profile)
            env_kwargs.update({
                "randomize_physics": True,
                "ramp_jitter": randomization_config.ramp_deg_max - randomization_config.ramp_deg_min,
                "friction_min": randomization_config.friction_min,
                "friction_max": randomization_config.friction_max,
                "randomization_profile": self.config.randomization_profile
            })
        
        # Create environment
        env = PassiveWalkerEnv(**env_kwargs)
        
        return env
    
    def collect_rollouts(self, env: PassiveWalkerEnv) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts with environment enhancements.
        
        Args:
            env: Enhanced environment
            
        Returns:
            Dictionary of collected experience
        """
        # Initialize rollout data
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        
        observations = np.zeros((self.config.n_steps, *obs_shape))
        actions = np.zeros((self.config.n_steps, *action_shape))
        rewards = np.zeros(self.config.n_steps)
        values = np.zeros(self.config.n_steps)
        log_probs = np.zeros(self.config.n_steps)
        dones = np.zeros(self.config.n_steps, dtype=bool)
        
        # Environment state
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        episode_success = False
        
        # Random number generator for perturbations
        rng = np.random.RandomState(42 + self.timestep)
        
        for step in range(self.config.n_steps):
            # Apply online perturbations
            if self.perturbation_manager is not None:
                perturbation_info = self.perturbation_manager.apply_perturbation(
                    env.model, env.data, step, rng
                )
                
                if perturbation_info["impulse_applied"]:
                    self.environment_stats["perturbation_applied"] += 1
            
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                
                if hasattr(self.model, 'get_action'):
                    action_output = self.model.get_action(obs_tensor)
                    if len(action_output) == 4:
                        # Temporal model (LSTM/GRU)
                        action, log_prob, value, _ = action_output
                    else:
                        # MLP model
                        action, log_prob, value = action_output
                else:
                    # MLP model
                    action, log_prob, value = self.model.get_action(obs_tensor)
                
                action = action.squeeze(0).cpu().numpy()
                log_prob = log_prob.squeeze(0).item()
                value = value.squeeze(0).item()
            
            # Store current observations
            observations[step] = obs
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Apply reward curriculum if enabled
            # The curriculum transitions from FSM to research mode rewards
            if self.reward_curriculum_manager is not None:
                current_mode = self.reward_curriculum_manager.get_current_mode(self.episode)
                # Update environment mode dynamically based on curriculum
                if env.mode != current_mode:
                    env.mode = current_mode
            # Note: We now use the environment's native rewards instead of overriding them
            
            # Store experience
            actions[step] = action
            rewards[step] = reward
            values[step] = value
            log_probs[step] = log_prob
            dones[step] = done
            
            # Update statistics
            episode_return += reward
            episode_length += 1
            
            # Update observations
            obs = next_obs
            
            # Handle episode end
            if done:
                # Determine episode success
                episode_success = episode_length > 50 and episode_return > 0
                
                # Update statistics
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                self.episode_success_rate.append(episode_success)
                self.episode += 1
                
                # Update adaptive randomization
                if self.randomization_manager is not None:
                    performance = episode_return if episode_success else 0.0
                    self.randomization_manager.update_performance(performance)
                    
                    # Get current randomization parameters
                    rand_params = self.randomization_manager.get_randomization_params()
                    self.environment_stats["randomization_strength"].append(rand_params["strength"])
                
                # Update reward curriculum
                if self.reward_curriculum_manager is not None:
                    self.reward_curriculum_manager.update_episode(self.episode)
                    current_stage = self.reward_curriculum_manager.get_current_mode(self.episode)
                    self.environment_stats["curriculum_stage"].append(current_stage)
                
                # Reset environment
                obs, _ = env.reset()
                episode_return = 0
                episode_length = 0
                episode_success = False
        
        # Convert to tensors
        return {
            "observations": torch.FloatTensor(observations).to(self.device),
            "actions": torch.FloatTensor(actions).to(self.device),
            "rewards": torch.FloatTensor(rewards).to(self.device),
            "values": torch.FloatTensor(values).to(self.device),
            "log_probs": torch.FloatTensor(log_probs).to(self.device),
            "dones": torch.BoolTensor(dones).to(self.device)
        }
    
    def _compute_fsm_reward(self, obs: np.ndarray, action: np.ndarray, 
                          next_obs: np.ndarray, done: bool) -> float:
        """
        Compute simplified FSM-style reward.
        
        Args:
            obs: Current observation
            next_obs: Next observation
            action: Action taken
            done: Whether episode ended
            
        Returns:
            Reward value
        """
        # Simple reward: encourage forward progress and stability
        forward_reward = next_obs[0] - obs[0]  # Forward progress
        stability_reward = -abs(next_obs[2])  # Penalize pitch deviation
        
        # Action penalty
        action_penalty = -0.001 * np.sum(action**2)
        
        # Survival bonus
        survival_bonus = 0.1 if not done else 0.0
        
        total_reward = forward_reward + stability_reward + action_penalty + survival_bonus
        
        return total_reward
    
    def train(self, env: Optional[PassiveWalkerEnv] = None) -> Dict[str, Any]:
        """
        Train enhanced PPO agent.
        
        Args:
            env: Environment to train on (if None, creates enhanced environment)
            
        Returns:
            Training results
        """
        print(f"Starting enhanced PPO training for {self.config.total_timesteps} timesteps")
        
        # Setup environment enhancements
        self.setup_environment_enhancements()
        
        # Create enhanced environment if not provided
        if env is None:
            env = self.create_enhanced_environment()
        
        start_time = time.time()
        
        while self.timestep < self.config.total_timesteps:
            # Collect rollouts with enhancements
            rollout_data = self.collect_rollouts(env)
            self.timestep += self.config.n_steps
            
            # Update policy
            update_stats = self.update(rollout_data)
            
            # Log training statistics
            if self.tracker:
                self.tracker.log_scalars({
                    "train/policy_loss": update_stats["policy_loss"],
                    "train/value_loss": update_stats["value_loss"],
                    "train/entropy_loss": update_stats["entropy_loss"],
                    "train/kl_divergence": update_stats["kl_divergence"],
                    "train/clip_fraction": update_stats["clip_fraction"],
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"]
                }, self.timestep)
                
                # Log environment statistics
                if self.environment_stats["randomization_strength"]:
                    self.tracker.log_scalar(
                        "env/randomization_strength",
                        self.environment_stats["randomization_strength"][-1],
                        self.timestep
                    )
                
                if self.environment_stats["curriculum_stage"]:
                    stage = self.environment_stats["curriculum_stage"][-1]
                    stage_value = 1.0 if stage == "research" else 0.0
                    self.tracker.log_scalar(
                        "env/curriculum_stage",
                        stage_value,
                        self.timestep
                    )
                
                self.tracker.log_scalar(
                    "env/perturbation_applied",
                    self.environment_stats["perturbation_applied"],
                    self.timestep
                )
            
            # Evaluation
            if self.timestep % self.config.eval_freq == 0:
                eval_stats = self.evaluate(env, self.config.n_eval_episodes)
                
                if self.tracker:
                    self.tracker.log_scalars({
                        "eval/return": eval_stats["eval_return"],
                        "eval/return_std": eval_stats["eval_return_std"],
                        "eval/length": eval_stats["eval_length"],
                        "eval/length_std": eval_stats["eval_length_std"]
                    }, self.timestep)
                
                # Save best model
                if eval_stats["eval_return"] > self.best_eval_return:
                    self.best_eval_return = eval_stats["eval_return"]
                    self.save_model("best_model.pth")
                
                print(f"Timestep {self.timestep}: Eval return = {eval_stats['eval_return']:.2f} ± {eval_stats['eval_return_std']:.2f}")
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if self.timestep % self.config.log_freq == 0:
                avg_return = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0
                avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                success_rate = np.mean(self.episode_success_rate[-100:]) if self.episode_success_rate else 0
                
                print(f"Timestep {self.timestep}: Avg return = {avg_return:.2f}, Avg length = {avg_length:.2f}, Success rate = {success_rate:.2f}")
        
        training_time = time.time() - start_time
        
        return {
            "training_time": training_time,
            "final_timestep": self.timestep,
            "best_eval_return": self.best_eval_return,
            "final_eval_stats": self.evaluate(env, self.config.n_eval_episodes),
            "environment_stats": self.environment_stats
        }
    
    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy using PPO with enhanced statistics.
        
        Args:
            rollout_data: Collected rollout data
            
        Returns:
            Training statistics
        """
        # Get next values for GAE computation
        with torch.no_grad():
            next_obs = rollout_data["observations"][-1].unsqueeze(0)
            if hasattr(self.model, 'forward'):
                forward_output = self.model.forward(next_obs)
                if len(forward_output) == 4:
                    # Temporal model (LSTM/GRU)
                    _, _, next_values, _ = forward_output
                else:
                    # MLP model
                    _, _, next_values = forward_output
            else:
                # MLP model
                _, _, next_values = self.model.forward(next_obs)
            next_values = next_values.squeeze().item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rollout_data["rewards"],
            rollout_data["values"],
            rollout_data["dones"],
            next_values
        )
        
        # Training statistics
        epoch_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "kl_divergence": [],
            "clip_fraction": []
        }
        
        # PPO epochs
        for epoch in range(self.config.n_epochs):
            # Sample mini-batches
            batch_size = len(rollout_data["observations"])
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = rollout_data["observations"][batch_indices]
                batch_actions = rollout_data["actions"][batch_indices]
                batch_old_log_probs = rollout_data["log_probs"][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Compute loss
                loss_dict = self.compute_ppo_loss(
                    batch_obs, batch_actions, batch_old_log_probs,
                    batch_advantages, batch_returns
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Store statistics
                for key in epoch_stats:
                    epoch_stats[key].append(loss_dict[key].item())
        
        # Compute average statistics
        avg_stats = {key: np.mean(values) for key, values in epoch_stats.items()}
        
        # Update training statistics
        for key in self.training_stats:
            self.training_stats[key].append(avg_stats[key])
        
        return avg_stats


def create_enhanced_ppo_trainer(model: nn.Module, config: PPOConfig, **kwargs) -> EnhancedPPOTrainer:
    """Create enhanced PPO trainer with default settings."""
    return EnhancedPPOTrainer(model, config, **kwargs)
