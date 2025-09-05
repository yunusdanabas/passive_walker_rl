"""
Memory-efficient rollout buffer for reinforcement learning.

Preallocates memory once and reuses it across episodes to eliminate
per-step allocations and improve performance.
"""

import numpy as np
from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RolloutBuffer:
    """
    Memory-efficient ring buffer for rollout data.
    
    Preallocates memory once and reuses it across episodes to avoid
    per-step allocations and improve performance.
    """
    
    def __init__(self, rollout_len: int, obs_dim: int = 11, act_dim: int = 3):
        """Initialize rollout buffer with preallocated memory."""
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Preallocate buffers (no per-step allocations)
        self.buffers = {
            "obs": np.empty((rollout_len, obs_dim), dtype=np.float32),
            "act": np.empty((rollout_len, act_dim), dtype=np.float32),
            "rew": np.empty((rollout_len,), dtype=np.float32),
            "done": np.empty((rollout_len,), dtype=np.bool_),
            "info": np.empty((rollout_len,), dtype=np.object_),
        }
        
        # Buffer state
        self.current_step = 0
        self.is_full = False
        
        # Streaming normalization stats (Welford's algorithm)
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_var = np.ones(obs_dim, dtype=np.float32)
        self.obs_count = 0
    
    def reset(self) -> None:
        """Reset the buffer for a new rollout."""
        self.current_step = 0
        self.is_full = False
    
    def add_step(self, obs: np.ndarray, act: np.ndarray, rew: float, 
                 done: bool, info: Dict[str, Any]) -> None:
        """Add step data to buffer (zero-copy, in-place writes)."""
        if self.current_step >= self.rollout_len:
            raise ValueError(f"Buffer full! Cannot add more than {self.rollout_len} steps")
        
        # Direct array writes (no allocations)
        self.buffers["obs"][self.current_step] = obs
        self.buffers["act"][self.current_step] = act
        self.buffers["rew"][self.current_step] = rew
        self.buffers["done"][self.current_step] = done
        self.buffers["info"][self.current_step] = info
        
        # Update streaming normalization stats
        self._update_normalization_stats(obs)
        
        self.current_step += 1
        if self.current_step >= self.rollout_len:
            self.is_full = True
    
    def _update_normalization_stats(self, obs: np.ndarray) -> None:
        """Update normalization stats using Welford's online algorithm."""
        self.obs_count += 1
        
        if self.obs_count == 1:
            self.obs_mean[:] = obs
            self.obs_var[:] = 0.0
        else:
            # Incremental mean and variance update
            delta = obs - self.obs_mean
            self.obs_mean += delta / self.obs_count
            delta2 = obs - self.obs_mean
            self.obs_var += delta * delta2
    
    def get_rollout(self) -> Dict[str, np.ndarray]:
        """Get current rollout data (returns copies)."""
        if self.current_step == 0:
            raise ValueError("No data in buffer")
        
        end_idx = self.current_step
        return {
            "obs": self.buffers["obs"][:end_idx].copy(),
            "act": self.buffers["act"][:end_idx].copy(),
            "rew": self.buffers["rew"][:end_idx].copy(),
            "done": self.buffers["done"][:end_idx].copy(),
            "info": self.buffers["info"][:end_idx].copy(),
        }
    
    def get_normalized_obs(self) -> np.ndarray:
        """Get normalized observations for current rollout."""
        if self.current_step == 0:
            raise ValueError("No data in buffer")
        
        end_idx = self.current_step
        obs_data = self.buffers["obs"][:end_idx]
        
        # Normalize using current stats
        if self.obs_count > 1:
            std = np.sqrt(self.obs_var / (self.obs_count - 1))
            std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
            return (obs_data - self.obs_mean) / std
        else:
            return obs_data
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get current normalization stats (mean, std, count)."""
        if self.obs_count > 1:
            std = np.sqrt(self.obs_var / (self.obs_count - 1))
            std = np.where(std < 1e-8, 1.0, std)
        else:
            std = np.ones_like(self.obs_mean)
        
        return self.obs_mean.copy(), std, self.obs_count
    
    def normalize_obs_inplace(self, obs: np.ndarray) -> None:
        """Normalize observation in place using current stats."""
        if self.obs_count > 1:
            std = np.sqrt(self.obs_var / (self.obs_count - 1))
            std = np.where(std < 1e-8, 1.0, std)
            obs[:] = (obs - self.obs_mean) / std
        # If count <= 1, no normalization needed
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics in bytes."""
        total_bytes = 0
        usage = {}
        
        for key, arr in self.buffers.items():
            bytes_used = arr.nbytes
            usage[key] = bytes_used
            total_bytes += bytes_used
        
        usage["total"] = total_bytes
        return usage


class MultiEnvRolloutBuffer:
    """Memory-efficient rollout buffer for multiple environments."""
    
    def __init__(self, num_envs: int, rollout_len: int, obs_dim: int = 11, act_dim: int = 3):
        """Initialize multi-environment rollout buffer."""
        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # Preallocate buffers for all environments
        self.buffers = {
            "obs": np.empty((num_envs, rollout_len, obs_dim), dtype=np.float32),
            "act": np.empty((num_envs, rollout_len, act_dim), dtype=np.float32),
            "rew": np.empty((num_envs, rollout_len), dtype=np.float32),
            "done": np.empty((num_envs, rollout_len), dtype=np.bool_),
            "info": np.empty((num_envs, rollout_len), dtype=np.object_),
        }
        
        # Per-environment state
        self.current_steps = np.zeros(num_envs, dtype=np.int32)
        self.is_full = np.zeros(num_envs, dtype=np.bool_)
        
        # Per-environment normalization stats
        self.obs_means = np.zeros((num_envs, obs_dim), dtype=np.float32)
        self.obs_vars = np.ones((num_envs, obs_dim), dtype=np.float32)
        self.obs_counts = np.zeros(num_envs, dtype=np.int32)
    
    def reset_env(self, env_idx: int) -> None:
        """Reset buffer for a specific environment."""
        self.current_steps[env_idx] = 0
        self.is_full[env_idx] = False
    
    def reset_all(self) -> None:
        """Reset all environment buffers."""
        self.current_steps.fill(0)
        self.is_full.fill(False)
    
    def add_step(self, env_idx: int, obs: np.ndarray, act: np.ndarray, 
                 rew: float, done: bool, info: Dict[str, Any]) -> None:
        """Add step data for specific environment (zero-copy)."""
        if self.current_steps[env_idx] >= self.rollout_len:
            raise ValueError(f"Buffer full for env {env_idx}")
        
        # Direct array writes (no allocations)
        step_idx = self.current_steps[env_idx]
        self.buffers["obs"][env_idx, step_idx] = obs
        self.buffers["act"][env_idx, step_idx] = act
        self.buffers["rew"][env_idx, step_idx] = rew
        self.buffers["done"][env_idx, step_idx] = done
        self.buffers["info"][env_idx, step_idx] = info
        
        # Update normalization stats
        self._update_normalization_stats(env_idx, obs)
        
        self.current_steps[env_idx] += 1
        if self.current_steps[env_idx] >= self.rollout_len:
            self.is_full[env_idx] = True
    
    def _update_normalization_stats(self, env_idx: int, obs: np.ndarray) -> None:
        """Update normalization stats for specific environment."""
        self.obs_counts[env_idx] += 1
        count = self.obs_counts[env_idx]
        
        if count == 1:
            self.obs_means[env_idx] = obs
            self.obs_vars[env_idx] = 0.0
        else:
            # Incremental mean and variance update
            delta = obs - self.obs_means[env_idx]
            self.obs_means[env_idx] += delta / count
            delta2 = obs - self.obs_means[env_idx]
            self.obs_vars[env_idx] += delta * delta2
    
    def get_rollout(self, env_idx: int) -> Dict[str, np.ndarray]:
        """Get rollout data for specific environment (returns copies)."""
        if self.current_steps[env_idx] == 0:
            raise ValueError(f"No data in buffer for env {env_idx}")
        
        end_idx = self.current_steps[env_idx]
        return {
            "obs": self.buffers["obs"][env_idx, :end_idx].copy(),
            "act": self.buffers["act"][env_idx, :end_idx].copy(),
            "rew": self.buffers["rew"][env_idx, :end_idx].copy(),
            "done": self.buffers["done"][env_idx, :end_idx].copy(),
            "info": self.buffers["info"][env_idx, :end_idx].copy(),
        }
    
    def get_all_rollouts(self) -> Dict[str, np.ndarray]:
        """Get rollout data for all environments (returns copies)."""
        return {
            "obs": self.buffers["obs"].copy(),
            "act": self.buffers["act"].copy(),
            "rew": self.buffers["rew"].copy(),
            "done": self.buffers["done"].copy(),
            "info": self.buffers["info"].copy(),
        }
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics in bytes."""
        total_bytes = 0
        usage = {}
        
        for key, arr in self.buffers.items():
            bytes_used = arr.nbytes
            usage[key] = bytes_used
            total_bytes += bytes_used
        
        usage["total"] = total_bytes
        return usage
