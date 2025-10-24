"""
PPO Rollout Buffer

Simple rollout buffer for PPO training with GAE computation.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import gymnasium as gym


class RolloutBuffer:
    """
    Simple rollout buffer for PPO.
    
    Stores experience and computes advantages using GAE.
    """
    
    def __init__(self, 
                 buffer_size: int,
                 obs_dim: int,
                 action_dim: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu"):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Size of buffer
            obs_dim: Observation dimension
            action_dim: Action dimension
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Buffer tensors
        self.observations = torch.zeros((buffer_size, obs_dim), device=device)
        self.actions = torch.zeros((buffer_size, action_dim), device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.values = torch.zeros(buffer_size, device=device)
        self.log_probs = torch.zeros(buffer_size, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
        # Computed values
        self.advantages = torch.zeros(buffer_size, device=device)
        self.returns = torch.zeros(buffer_size, device=device)
        
        # Buffer state
        self.ptr = 0
        self.full = False
    
    def add(self, 
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            value: float,
            log_prob: float,
            done: bool):
        """
        Add experience to buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
        """
        # Convert to tensors if needed
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(self.device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(self.device)
        
        # Store experience
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = bool(done)
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.ptr = 0
            self.full = True
    
    def add_batch(self,
                  obs: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  values: np.ndarray,
                  log_probs: np.ndarray,
                  dones: np.ndarray):
        """
        Add batch of experiences to buffer.
        
        Args:
            obs: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            values: Batch of value estimates
            log_probs: Batch of log probabilities
            dones: Batch of done flags
        """
        batch_size = len(obs)
        
        for i in range(batch_size):
            self.add(
                obs[i], actions[i], rewards[i], 
                values[i], log_probs[i], dones[i]
            )
    
    def compute_gae(self, next_value: float = 0.0):
        """
        Compute GAE advantages and returns.
        
        Args:
            next_value: Value estimate for next state
        """
        # Convert next_value to tensor
        if isinstance(next_value, (int, float)):
            next_value = torch.tensor(next_value, device=self.device)
        
        # Get buffer size
        buffer_size = self.buffer_size if self.full else self.ptr
        
        # Initialize advantage and return arrays
        advantages = torch.zeros(buffer_size, device=self.device)
        returns = torch.zeros(buffer_size, device=self.device)
        
        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(buffer_size)):
            if t == buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_value_t = self.values[t + 1]
            
            # Compute TD error
            delta = self.rewards[t] + self.gamma * next_value_t * next_non_terminal - self.values[t]
            
            # Compute GAE
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
        
        # Store computed values
        self.advantages[:buffer_size] = advantages
        self.returns[:buffer_size] = returns
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from buffer.
        
        Returns:
            Dictionary of buffer data
        """
        buffer_size = self.buffer_size if self.full else self.ptr
        
        return {
            "observations": self.observations[:buffer_size],
            "actions": self.actions[:buffer_size],
            "rewards": self.rewards[:buffer_size],
            "values": self.values[:buffer_size],
            "log_probs": self.log_probs[:buffer_size],
            "advantages": self.advantages[:buffer_size],
            "returns": self.returns[:buffer_size],
            "dones": self.dones[:buffer_size]
        }
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary of sampled data
        """
        buffer_size = self.buffer_size if self.full else self.ptr
        
        # Generate random indices
        indices = torch.randint(0, buffer_size, (batch_size,), device=self.device)
        
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "values": self.values[indices],
            "log_probs": self.log_probs[indices],
            "advantages": self.advantages[indices],
            "returns": self.returns[indices],
            "dones": self.dones[indices]
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.full = False
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.full
    
    def size(self) -> int:
        """Get current buffer size."""
        return self.buffer_size if self.full else self.ptr


class VectorizedRolloutBuffer:
    """
    Vectorized rollout buffer for multiple environments.
    
    Manages multiple RolloutBuffer instances for parallel environments.
    """
    
    def __init__(self, 
                 buffer_size: int,
                 obs_dim: int,
                 action_dim: int,
                 n_envs: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: str = "cpu"):
        """
        Initialize vectorized rollout buffer.
        
        Args:
            buffer_size: Size of buffer per environment
            obs_dim: Observation dimension
            action_dim: Action dimension
            n_envs: Number of environments
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = device
        
        # Create individual buffers for each environment
        self.buffers = [
            RolloutBuffer(
                buffer_size, obs_dim, action_dim,
                gamma, gae_lambda, device
            )
            for _ in range(n_envs)
        ]
        
        # Track current environment
        self.current_env = 0
    
    def add(self, 
            obs: np.ndarray,
            action: np.ndarray,
            reward: float,
            value: float,
            log_prob: float,
            done: bool,
            env_idx: Optional[int] = None):
        """
        Add experience to buffer.
        
        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode ended
            env_idx: Environment index (if None, uses current_env)
        """
        if env_idx is None:
            env_idx = self.current_env
        
        self.buffers[env_idx].add(obs, action, reward, value, log_prob, done)
    
    def add_batch(self,
                  obs: np.ndarray,
                  actions: np.ndarray,
                  rewards: np.ndarray,
                  values: np.ndarray,
                  log_probs: np.ndarray,
                  dones: np.ndarray):
        """
        Add batch of experiences to all buffers.
        
        Args:
            obs: Batch of observations (n_envs, obs_dim)
            actions: Batch of actions (n_envs, action_dim)
            rewards: Batch of rewards (n_envs,)
            values: Batch of value estimates (n_envs,)
            log_probs: Batch of log probabilities (n_envs,)
            dones: Batch of done flags (n_envs,)
        """
        for env_idx in range(self.n_envs):
            self.buffers[env_idx].add(
                obs[env_idx], actions[env_idx], rewards[env_idx],
                values[env_idx], log_probs[env_idx], dones[env_idx]
            )
    
    def compute_gae(self, next_values: np.ndarray):
        """
        Compute GAE advantages and returns for all buffers.
        
        Args:
            next_values: Value estimates for next states (n_envs,)
        """
        for env_idx in range(self.n_envs):
            self.buffers[env_idx].compute_gae(next_values[env_idx])
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from all buffers.
        
        Returns:
            Dictionary of concatenated buffer data
        """
        # Collect data from all buffers
        all_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "advantages": [],
            "returns": [],
            "dones": []
        }
        
        for buffer in self.buffers:
            data = buffer.get()
            for key in all_data:
                all_data[key].append(data[key])
        
        # Concatenate all data
        for key in all_data:
            all_data[key] = torch.cat(all_data[key], dim=0)
        
        return all_data
    
    def sample_all(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch from all buffers.
        
        Args:
            batch_size: Size of batch to sample
            
        Returns:
            Dictionary of sampled data
        """
        # Collect samples from all buffers
        all_samples = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "advantages": [],
            "returns": [],
            "dones": []
        }
        
        # Sample from each buffer
        samples_per_env = batch_size // self.n_envs
        remaining_samples = batch_size % self.n_envs
        
        for i, buffer in enumerate(self.buffers):
            # Determine sample size for this environment
            env_batch_size = samples_per_env
            if i < remaining_samples:
                env_batch_size += 1
            
            if env_batch_size > 0:
                sample = buffer.sample(env_batch_size)
                for key in all_samples:
                    all_samples[key].append(sample[key])
        
        # Concatenate all samples
        for key in all_samples:
            if all_samples[key]:
                all_samples[key] = torch.cat(all_samples[key], dim=0)
            else:
                # Handle case where no samples were collected
                all_samples[key] = torch.empty(0, device=self.device)
        
        return all_samples
    
    def clear_all(self):
        """Clear all buffers."""
        for buffer in self.buffers:
            buffer.clear()
    
    def all_full(self) -> bool:
        """Check if all buffers are full."""
        return all(buffer.is_full() for buffer in self.buffers)
    
    def total_size(self) -> int:
        """Get total size across all buffers."""
        return sum(buffer.size() for buffer in self.buffers)
