"""
Rollout buffer for storing and managing environment trajectories.

- Preallocated arrays for zero-copy data collection
- Streaming normalization statistics for observations
- Support for single and multi-environment collection
- Save/load functionality for BC and PPO training
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os


class RolloutBuffer:
    """
    Single-environment rollout buffer with preallocated arrays and streaming normalization.

    Stores (obs, act, rew, done, info) + optional extras from reward functions.
    Maintains streaming normalization stats for observations using Welford's algorithm.
    """

    def __init__(self, rollout_len: int, obs_dim: int, act_dim: int, store_extras: bool = True):
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.store_extras = store_extras

        # Preallocated arrays
        self.obs = np.zeros((rollout_len, obs_dim), dtype=np.float32)
        self.act = np.zeros((rollout_len, act_dim), dtype=np.float32)
        self.rew = np.zeros(rollout_len, dtype=np.float32)
        self.done = np.zeros(rollout_len, dtype=bool)
        self.info = np.empty(rollout_len, dtype=object)

        # Optional extras storage (reward breakdown)
        if store_extras:
            self.extras = {}
            # Common reward keys from our reward system
            self.extras_keys = [
                "r_forward",
                "r_upright",
                "r_vel",
                "r_sym",
                "r_clear",
                "r_act_cost",
                "fell",
            ]
            for key in self.extras_keys:
                self.extras[key] = np.zeros(rollout_len, dtype=np.float32)
        else:
            self.extras = None

        # Streaming normalization stats (Welford's algorithm)
        self.norm_mean = np.zeros(obs_dim, dtype=np.float64)
        self.norm_M2 = np.zeros(obs_dim, dtype=np.float64)
        self.norm_count = 0

        # Current position
        self.pos = 0

    def reset(self) -> None:
        """Reset buffer to empty state."""
        self.pos = 0
        self.norm_mean.fill(0.0)
        self.norm_M2.fill(0.0)
        self.norm_count = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: Dict[str, Any],
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add one step to buffer. Raises if over capacity."""
        if self.pos >= self.rollout_len:
            raise RuntimeError(f"Buffer overflow: {self.pos} >= {self.rollout_len}")

        # Store data
        self.obs[self.pos] = obs
        self.act[self.pos] = act
        self.rew[self.pos] = rew
        self.done[self.pos] = done
        self.info[self.pos] = info

        # Update streaming normalization stats
        self._update_norm_stats(obs)

        # Store extras if enabled
        if self.store_extras and extras is not None:
            for key in self.extras_keys:
                if key in extras:
                    self.extras[key][self.pos] = extras[key]
                else:
                    self.extras[key][self.pos] = 0.0

        self.pos += 1

    def _update_norm_stats(self, obs: np.ndarray) -> None:
        """Update streaming normalization statistics using Welford's algorithm."""
        self.norm_count += 1
        delta = obs - self.norm_mean
        self.norm_mean += delta / self.norm_count
        delta2 = obs - self.norm_mean
        self.norm_M2 += delta * delta2

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.pos >= self.rollout_len

    def size(self) -> int:
        """Get current number of stored steps."""
        return self.pos

    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data as copies, trimmed to actual size."""
        size = self.size()
        result = {
            "obs": self.obs[:size].copy(),
            "act": self.act[:size].copy(),
            "rew": self.rew[:size].copy(),
            "done": self.done[:size].copy(),
            "info": self.info[:size].copy(),
        }

        if self.store_extras and self.extras is not None:
            result["extras"] = {key: self.extras[key][:size].copy() for key in self.extras_keys}

        return result

    def get_normalized_obs(self) -> np.ndarray:
        """Get observations normalized using frozen stats."""
        size = self.size()
        if size == 0:
            return np.array([])

        # Compute std from M2 and count
        if self.norm_count <= 1:
            std = np.ones(self.obs_dim, dtype=np.float32)
        else:
            variance = self.norm_M2 / (self.norm_count - 1)
            std = np.sqrt(variance)
            std = np.maximum(std, 1e-8)  # Avoid division by zero

        # Normalize
        normalized = (self.obs[:size] - self.norm_mean) / std
        return normalized.astype(np.float32)

    def get_norm_stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get normalization statistics: (mean, std, count)."""
        if self.norm_count <= 1:
            std = np.ones(self.obs_dim, dtype=np.float32)
        else:
            variance = self.norm_M2 / (self.norm_count - 1)
            std = np.sqrt(variance)
            std = np.maximum(std, 1e-8)

        return self.norm_mean.astype(np.float32), std.astype(np.float32), self.norm_count

    def save_npz(self, path: str) -> None:
        """Save buffer data to NPZ file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = self.get()

        # Convert info array to list for serialization
        data["info"] = [info for info in data["info"]]

        # Add normalization stats
        mean, std, count = self.get_norm_stats()
        data["norm_mean"] = mean
        data["norm_std"] = std
        data["norm_count"] = count

        # Add metadata
        data["metadata"] = {
            "rollout_len": self.rollout_len,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "store_extras": self.store_extras,
            "size": self.size(),
        }

        # Flatten extras dict for NPZ
        if "extras" in data:
            extras = data.pop("extras")
            for key, value in extras.items():
                data[f"extras_{key}"] = value

        np.savez(path, **data)

    @staticmethod
    def load_npz(path: str) -> Dict[str, Any]:
        """Load buffer data from NPZ file."""
        data = np.load(path, allow_pickle=True)

        # Convert back to dict
        result = {key: data[key] for key in data.keys()}

        # Convert info back to object array
        if "info" in result:
            result["info"] = np.array(result["info"], dtype=object)

        # Reconstruct extras dict
        extras = {}
        extras_keys = ["r_forward", "r_upright", "r_vel", "r_sym", "r_clear", "r_act_cost", "fell"]
        for key in extras_keys:
            extras_key = f"extras_{key}"
            if extras_key in result:
                extras[key] = result.pop(extras_key)

        if extras:
            result["extras"] = extras

        return result


class MultiEnvRolloutBuffer:
    """
    Multi-environment rollout buffer for vectorized collection.

    Manages multiple RolloutBuffer instances, one per environment.
    Provides convenient access to individual buffers and stacked data.
    """

    def __init__(
        self, num_envs: int, rollout_len: int, obs_dim: int, act_dim: int, store_extras: bool = True
    ):
        self.num_envs = num_envs
        self.buffers = [
            RolloutBuffer(rollout_len, obs_dim, act_dim, store_extras) for _ in range(num_envs)
        ]

    def reset_env(self, i: int) -> None:
        """Reset buffer for environment i."""
        if 0 <= i < self.num_envs:
            self.buffers[i].reset()
        else:
            raise IndexError(f"Environment index {i} out of range [0, {self.num_envs})")

    def add(
        self,
        i: int,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        done: bool,
        info: Dict[str, Any],
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add step to environment i's buffer."""
        if 0 <= i < self.num_envs:
            self.buffers[i].add(obs, act, rew, done, info, extras)
        else:
            raise IndexError(f"Environment index {i} out of range [0, {self.num_envs})")

    def is_full(self, i: int) -> bool:
        """Check if environment i's buffer is full."""
        if 0 <= i < self.num_envs:
            return self.buffers[i].is_full()
        else:
            raise IndexError(f"Environment index {i} out of range [0, {self.num_envs})")

    def get(self, i: int) -> Dict[str, np.ndarray]:
        """Get data from environment i's buffer."""
        if 0 <= i < self.num_envs:
            return self.buffers[i].get()
        else:
            raise IndexError(f"Environment index {i} out of range [0, {self.num_envs})")

    def stacked(self) -> Dict[str, np.ndarray]:
        """Return stacked arrays from all environments, aligned to minimum size."""
        if not self.buffers:
            return {}

        # Find minimum size across all buffers
        min_size = min(buf.size() for buf in self.buffers)
        if min_size == 0:
            return {}

        # Stack arrays
        result = {}
        for key in ["obs", "act", "rew", "done"]:
            arrays = [buf.get()[key][:min_size] for buf in self.buffers]
            result[key] = np.stack(arrays, axis=0)  # (num_envs, min_size, ...)

        # Handle info separately (object arrays)
        info_arrays = [buf.get()["info"][:min_size] for buf in self.buffers]
        result["info"] = np.stack(info_arrays, axis=0)

        # Handle extras if present
        if self.buffers[0].store_extras and self.buffers[0].extras is not None:
            result["extras"] = {}
            for key in self.buffers[0].extras_keys:
                arrays = [buf.get()["extras"][key][:min_size] for buf in self.buffers]
                result["extras"][key] = np.stack(arrays, axis=0)

        return result
