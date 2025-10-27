"""
Vectorized Environment Wrapper

Simple vectorized environment wrapper for parallel PPO training.
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from typing import Callable, List, Dict, Any, Optional
import numpy as np
import torch
import time


class VectorizedEnvWrapper:
    """
    Simple wrapper for vectorized environments.
    
    Handles parallel environment execution for PPO training.
    """
    
    def __init__(self, 
                 env_fns: List[Callable],
                 vectorization_type: str = "sync",
                 **kwargs):
        """
        Initialize vectorized environment wrapper.
        
        Args:
            env_fns: List of environment factory functions
            vectorization_type: Type of vectorization ("sync" or "async")
            **kwargs: Additional arguments for vectorized environment
        """
        self.env_fns = env_fns
        self.n_envs = len(env_fns)
        
        # Create vectorized environment
        if vectorization_type == "sync":
            self.env = SyncVectorEnv(env_fns, **kwargs)
        elif vectorization_type == "async":
            self.env = AsyncVectorEnv(env_fns, **kwargs)
        else:
            raise ValueError(f"Unknown vectorization type: {vectorization_type}")
        
        self.vectorization_type = vectorization_type
        
        # Environment properties
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    
    def reset(self, **kwargs) -> tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all environments.
        
        Args:
            **kwargs: Additional arguments for reset
            
        Returns:
            observations: Initial observations from all environments
            infos: Additional information from all environments
        """
        return self.env.reset(**kwargs)
    
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            actions: Actions for all environments
            
        Returns:
            observations: Next observations
            rewards: Rewards received
            terminated: Termination flags
            truncated: Truncation flags
            infos: Additional information
        """
        return self.env.step(actions)
    
    def close(self):
        """Close all environments."""
        self.env.close()
    
    def render(self, **kwargs):
        """Render environments."""
        return self.env.render(**kwargs)


def create_vectorized_env(env_name: str, 
                        n_envs: int,
                        env_kwargs: Dict[str, Any] = None,
                        vectorization_type: str = "sync") -> VectorizedEnvWrapper:
    """
    Create vectorized environment.
    
    Args:
        env_name: Name of environment class
        n_envs: Number of parallel environments
        env_kwargs: Environment arguments
        vectorization_type: Type of vectorization
        
    Returns:
        Vectorized environment wrapper
    """
    if env_kwargs is None:
        env_kwargs = {}
    
    # Create environment factory functions
    env_fns = []
    for i in range(n_envs):
        def make_env(env_id=i):
            # Import environment class
            if env_name == "PassiveWalkerEnv":
                from ..core.env import PassiveWalkerEnv
                return PassiveWalkerEnv(**env_kwargs)
            else:
                return gym.make(env_name, **env_kwargs)
        
        env_fns.append(make_env)
    
    return VectorizedEnvWrapper(env_fns, vectorization_type)


class PPOVectorizedTrainer:
    """
    PPO trainer for vectorized environments.
    
    Handles parallel rollout collection and training.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: PPOConfig,
                 device: str = "cpu",
                 tracker: Optional[Any] = None):
        """
        Initialize vectorized PPO trainer.
        
        Args:
            model: Actor-critic model
            config: PPO configuration
            device: Device to train on
            tracker: Experiment tracker
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.tracker = tracker
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
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
    
    def collect_rollouts(self, env: VectorizedEnvWrapper) -> Dict[str, torch.Tensor]:
        """
        Collect rollouts from vectorized environment.
        
        Args:
            env: Vectorized environment
            
        Returns:
            Dictionary of collected experience
        """
        # Initialize rollout data
        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        
        observations = np.zeros((self.config.n_steps, env.n_envs, *obs_shape))
        actions = np.zeros((self.config.n_steps, env.n_envs, *action_shape))
        rewards = np.zeros((self.config.n_steps, env.n_envs))
        values = np.zeros((self.config.n_steps, env.n_envs))
        log_probs = np.zeros((self.config.n_steps, env.n_envs))
        dones = np.zeros((self.config.n_steps, env.n_envs), dtype=bool)
        
        # Reset environments
        obs, _ = env.reset()
        
        # Collect rollouts
        for step in range(self.config.n_steps):
            # Get actions from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(self.device)
                
                if hasattr(self.model, 'get_action'):
                    # Temporal model
                    action, log_prob, value = self.model.get_action(obs_tensor)
                else:
                    # MLP model
                    action, log_prob, value = self.model.get_action(obs_tensor)
                
                action = action.cpu().numpy()
                log_prob = log_prob.cpu().numpy()
                value = value.cpu().numpy()
            
            # Store current observations
            observations[step] = obs
            
            # Take actions in environments
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            
            # Store experience
            actions[step] = action
            rewards[step] = reward
            values[step] = value.squeeze()
            log_probs[step] = log_prob.squeeze()
            dones[step] = done
            
            # Update observations
            obs = next_obs
            
            # Track episode statistics
            for env_idx in range(env.n_envs):
                if done[env_idx]:
                    self.episode += 1
        
        # Convert to tensors
        return {
            "observations": torch.FloatTensor(observations).to(self.device),
            "actions": torch.FloatTensor(actions).to(self.device),
            "rewards": torch.FloatTensor(rewards).to(self.device),
            "values": torch.FloatTensor(values).to(self.device),
            "log_probs": torch.FloatTensor(log_probs).to(self.device),
            "dones": torch.BoolTensor(dones).to(self.device)
        }
    
    def compute_gae(self, 
                   rewards: torch.Tensor,
                   values: torch.Tensor,
                   dones: torch.Tensor,
                   next_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE for vectorized rollouts.
        
        Args:
            rewards: Rewards (n_steps, n_envs)
            values: Value estimates (n_steps, n_envs)
            dones: Done flags (n_steps, n_envs)
            next_values: Next value estimates (n_envs,)
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages for each environment
        for env_idx in range(rewards.shape[1]):
            env_rewards = rewards[:, env_idx]
            env_values = values[:, env_idx]
            env_dones = dones[:, env_idx]
            next_value = next_values[env_idx]
            
            # Compute GAE for this environment
            gae = 0.0
            for t in reversed(range(len(env_rewards))):
                if t == len(env_rewards) - 1:
                    next_non_terminal = 1.0 - env_dones[t].float()
                    next_value_t = next_value
                else:
                    next_non_terminal = 1.0 - env_dones[t].float()
                    next_value_t = env_values[t + 1]
                
                # Compute TD error
                delta = env_rewards[t] + self.config.gamma * next_value_t * next_non_terminal - env_values[t]
                
                # Compute GAE
                gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
                
                advantages[t, env_idx] = gae
                returns[t, env_idx] = advantages[t, env_idx] + env_values[t]
        
        return advantages, returns
    
    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            rollout_data: Collected rollout data
            
        Returns:
            Training statistics
        """
        # Get next values for GAE computation
        with torch.no_grad():
            next_obs = rollout_data["observations"][-1]  # Last observations
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
            next_values = next_values.squeeze()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rollout_data["rewards"],
            rollout_data["values"],
            rollout_data["dones"],
            next_values
        )
        
        # Flatten data for training
        obs_flat = rollout_data["observations"].reshape(-1, *rollout_data["observations"].shape[2:])
        actions_flat = rollout_data["actions"].reshape(-1, *rollout_data["actions"].shape[2:])
        old_log_probs_flat = rollout_data["log_probs"].reshape(-1)
        advantages_flat = advantages.reshape(-1)
        returns_flat = returns.reshape(-1)
        
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
            batch_size = len(obs_flat)
            indices = torch.randperm(batch_size)
            
            for start_idx in range(0, batch_size, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = obs_flat[batch_indices]
                batch_actions = actions_flat[batch_indices]
                batch_old_log_probs = old_log_probs_flat[batch_indices]
                batch_advantages = advantages_flat[batch_indices]
                batch_returns = returns_flat[batch_indices]
                
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
        return {key: np.mean(values) for key, values in epoch_stats.items()}
    
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
            # Temporal model
            log_probs, entropy, values = self.model.evaluate_actions(obs, actions)
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
    
    def train(self, env: VectorizedEnvWrapper) -> Dict[str, Any]:
        """
        Train PPO agent with vectorized environments.
        
        Args:
            env: Vectorized environment
            
        Returns:
            Training results
        """
        print(f"Starting vectorized PPO training for {self.config.total_timesteps} timesteps")
        
        start_time = time.time()
        
        while self.timestep < self.config.total_timesteps:
            # Collect rollouts
            rollout_data = self.collect_rollouts(env)
            self.timestep += self.config.n_steps * env.n_envs
            
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
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            if self.timestep % self.config.log_freq == 0:
                print(f"Timestep {self.timestep}: Policy loss = {update_stats['policy_loss']:.4f}, Value loss = {update_stats['value_loss']:.4f}")
        
        training_time = time.time() - start_time
        
        return {
            "training_time": training_time,
            "final_timestep": self.timestep,
            "best_eval_return": self.best_eval_return
        }


def create_vectorized_ppo_trainer(model: nn.Module, config: PPOConfig, **kwargs) -> PPOVectorizedTrainer:
    """Create vectorized PPO trainer with default settings."""
    return PPOVectorizedTrainer(model, config, **kwargs)
