"""
PPO Training Pipeline

Complete PPO implementation with proper loss computation and environment integration.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import time
import os

from .models import create_actor_critic, load_bc_weights
from .config import PPOConfig
from .buffer import RolloutBuffer, VectorizedRolloutBuffer
from ..bc.experiment.tracking import ExperimentTracker


class PPOTrainer:
    """
    Simple PPO trainer with proper loss computation.
    
    Handles rollout collection, advantage computation, and policy updates.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: PPOConfig,
                 device: str = "cpu",
                 tracker: Optional[ExperimentTracker] = None,
                 output_dir: str = "ppo_runs"):
        """
        Initialize PPO trainer.
        
        Args:
            model: Actor-critic model
            config: PPO configuration
            device: Device to train on
            tracker: Experiment tracker for logging
            output_dir: Directory to save outputs
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tracker = tracker
        self.output_dir = output_dir
        self.run_dir = None  # Will be set when saving
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=config.total_timesteps // config.n_steps
        )
        
        # Training state
        self.timestep = 0
        self.episode = 0
        self.best_eval_return = float('-inf')
        
        # Statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "kl_divergence": [],
            "clip_fraction": []
        }
    
    def compute_gae(self, 
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   dones: torch.Tensor,
                   next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards (n_steps,)
            values: Value estimates (n_steps,)
            dones: Done flags (n_steps,)
            next_value: Value estimate for next state
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value_t = values[t + 1]
            
            # Compute TD error
            delta = rewards[t] + self.config.gamma * next_value_t * next_non_terminal - values[t]
            
            # Compute GAE
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def compute_ppo_loss(self,
                        obs: torch.Tensor,
                        actions: torch.Tensor,
                        old_log_probs: torch.Tensor,
                        advantages: torch.Tensor,
                        returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss with proper clipping.
        
        Args:
            obs: Observations
            actions: Actions taken
            old_log_probs: Old log probabilities
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Dictionary of loss components
        """
        # Get current policy outputs
        if hasattr(self.model, 'evaluate_actions'):
            eval_output = self.model.evaluate_actions(obs, actions)
            if len(eval_output) == 4:
                # Temporal model (LSTM/GRU)
                log_probs, entropy, values, _ = eval_output
            else:
                # MLP model
                log_probs, entropy, values = eval_output
        else:
            # MLP model
            log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy ratio
        ratio = torch.exp(log_probs.squeeze() - old_log_probs)
        
        # Compute clipped policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_pred_clipped = old_log_probs + torch.clamp(
            values.squeeze() - old_log_probs,
            -self.config.clip_range,
            self.config.clip_range
        )
        value_loss1 = (values.squeeze() - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        value_loss = torch.max(value_loss1, value_loss2).mean()
        
        # Compute entropy loss
        entropy_loss = -entropy.mean()
        
        # Compute total loss
        total_loss = (
            policy_loss + 
            self.config.value_loss_coef * value_loss + 
            self.config.entropy_coef * entropy_loss
        )
        
        # Compute additional statistics
        with torch.no_grad():
            kl_divergence = (old_log_probs - log_probs.squeeze()).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_range).float().mean()
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "kl_divergence": kl_divergence,
            "clip_fraction": clip_fraction
        }
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            buffer: Rollout buffer with collected experience
            
        Returns:
            Dictionary of training statistics
        """
        # Get all data from buffer
        data = buffer.get()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            data["rewards"],
            data["values"],
            data["dones"]
        )
        
        # Store advantages and returns in buffer
        buffer.advantages[:len(advantages)] = advantages
        buffer.returns[:len(returns)] = returns
        
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
            batch_size = len(data["observations"])
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = data["observations"][batch_indices]
                batch_actions = data["actions"][batch_indices]
                batch_old_log_probs = data["log_probs"][batch_indices]
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
    
    def collect_rollouts(self, env: gym.Env) -> RolloutBuffer:
        """
        Collect rollouts from environment.
        
        Args:
            env: Environment to collect from
            
        Returns:
            Rollout buffer with collected experience
        """
        buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            device=self.device
        )
        
        obs, _ = env.reset()
        episode_return = 0
        episode_length = 0
        
        for step in range(self.config.n_steps):
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
            
            # Take action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            buffer.add(obs, action, reward, value, log_prob, done)
            
            # Update statistics
            episode_return += reward
            episode_length += 1
            
            # Update observation
            obs = next_obs
            
            # Handle episode end
            if done:
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                self.episode += 1
                
                # Reset environment
                obs, _ = env.reset()
                episode_return = 0
                episode_length = 0
        
        return buffer
    
    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate current policy.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_returns = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0
            episode_length = 0
            
            while True:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    if hasattr(self.model, 'get_action'):
                        action_output = self.model.get_action(obs_tensor, deterministic=True)
                        if len(action_output) == 4:
                            # Temporal model (LSTM/GRU)
                            action, _, _, _ = action_output
                        else:
                            # MLP model
                            action, _, _ = action_output
                    else:
                        # MLP model
                        action, _, _ = self.model.get_action(obs_tensor, deterministic=True)
                    
                    action = action.squeeze(0).cpu().numpy()
                
                obs, reward, done, info = env.step(action)
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        return {
            "eval_return": np.mean(episode_returns),
            "eval_return_std": np.std(episode_returns),
            "eval_length": np.mean(episode_lengths),
            "eval_length_std": np.std(episode_lengths)
        }
    
    def train(self, env: gym.Env) -> Dict[str, Any]:
        """
        Train PPO agent.
        
        Args:
            env: Environment to train on
            
        Returns:
            Training results
        """
        print(f"Starting PPO training for {self.config.total_timesteps} timesteps")
        
        start_time = time.time()
        
        while self.timestep < self.config.total_timesteps:
            # Collect rollouts
            buffer = self.collect_rollouts(env)
            self.timestep += self.config.n_steps
            
            # Update policy
            update_stats = self.update(buffer)
            
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
                
                print(f"Timestep {self.timestep}: Avg return = {avg_return:.2f}, Avg length = {avg_length:.2f}")
        
        training_time = time.time() - start_time
        
        return {
            "training_time": training_time,
            "final_timestep": self.timestep,
            "best_eval_return": self.best_eval_return,
            "final_eval_stats": self.evaluate(env, self.config.n_eval_episodes)
        }
    
    def save_model(self, filename: str = "model.pth"):
        """Save model checkpoint."""
        import os
        from datetime import datetime
        
        if not self.run_dir:
            # Use tracker's log dir if available, otherwise create one
            if self.tracker and hasattr(self.tracker, 'log_dir'):
                self.run_dir = self.tracker.log_dir
            else:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.run_dir = os.path.join(self.output_dir, f"{self.config.experiment_name}_{timestamp}")
                os.makedirs(self.run_dir, exist_ok=True)
        
        path = os.path.join(self.run_dir, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "timestep": self.timestep,
            "episode": self.episode,
            "best_eval_return": self.best_eval_return,
            "config": self.config.to_dict()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.timestep = checkpoint["timestep"]
        self.episode = checkpoint["episode"]
        self.best_eval_return = checkpoint["best_eval_return"]


def create_ppo_trainer(model: nn.Module, config: PPOConfig, **kwargs) -> PPOTrainer:
    """Create PPO trainer with default settings."""
    return PPOTrainer(model, config, **kwargs)
