"""
Data Augmentation for BC Training

Online augmentation during training to improve robustness.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import random
 


class BaseAugmentation:
    """Base class for data augmentation."""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to observation and action."""
        if random.random() < self.probability:
            return self._augment(obs, action)
        return obs, action
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement specific augmentation."""
        raise NotImplementedError


class ObservationNoise(BaseAugmentation):
    """Add Gaussian noise to observations."""
    
    def __init__(self, position_std: float = 0.01, velocity_std: float = 0.02, probability: float = 0.5):
        super().__init__(probability)
        self.position_std = position_std
        self.velocity_std = velocity_std
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to observations."""
        noisy_obs = obs.copy()
        
        # Position noise (indices 0, 1, 2, 5, 6, 7)
        position_indices = [0, 1, 2, 5, 6, 7]
        for i in position_indices:
            if i < len(noisy_obs):
                noise = np.random.normal(0, self.position_std)
                noisy_obs[i] += noise * abs(obs[i]) if obs[i] != 0 else noise * 0.01
        
        # Velocity noise (indices 3, 4, 8, 9, 10)
        velocity_indices = [3, 4, 8, 9, 10]
        for i in velocity_indices:
            if i < len(noisy_obs):
                noise = np.random.normal(0, self.velocity_std)
                noisy_obs[i] += noise * abs(obs[i]) if obs[i] != 0 else noise * 0.01
        
        return noisy_obs, action


class ActionNoise(BaseAugmentation):
    """Add small noise to actions."""
    
    def __init__(self, std: float = 0.01, probability: float = 0.3):
        super().__init__(probability)
        self.std = std
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to actions."""
        noisy_action = action.copy()
        noise = np.random.normal(0, self.std, size=action.shape)
        noisy_action += noise
        
        # Clip to action space bounds
        noisy_action = np.clip(noisy_action, -1.0, 1.0)
        
        return obs, noisy_action


class TemporalShift(BaseAugmentation):
    """Shift observations temporally (simulate timing variations)."""
    
    def __init__(self, max_shift: float = 0.1, probability: float = 0.2):
        super().__init__(probability)
        self.max_shift = max_shift
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal shift to observations."""
        # Simple implementation: scale velocity components
        shift_factor = 1.0 + np.random.uniform(-self.max_shift, self.max_shift)
        
        augmented_obs = obs.copy()
        velocity_indices = [3, 4, 8, 9, 10]  # velocity components
        for i in velocity_indices:
            if i < len(augmented_obs):
                augmented_obs[i] *= shift_factor
        
        return augmented_obs, action




class CompositeAugmentation:
    """Combine multiple augmentations."""
    
    def __init__(self, augmentations: list[BaseAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all augmentations in sequence."""
        current_obs, current_action = obs, action
        
        for aug in self.augmentations:
            current_obs, current_action = aug(current_obs, current_action)
        
        return current_obs, current_action


def create_default_augmentation() -> CompositeAugmentation:
    """Create default augmentation pipeline."""
    return CompositeAugmentation([
        ObservationNoise(position_std=0.01, velocity_std=0.02, probability=0.5),
        ActionNoise(std=0.01, probability=0.3),
        TemporalShift(max_shift=0.1, probability=0.2),
    ])


def create_light_augmentation() -> CompositeAugmentation:
    """Create light augmentation pipeline."""
    return CompositeAugmentation([
        ObservationNoise(position_std=0.005, velocity_std=0.01, probability=0.3),
        ActionNoise(std=0.005, probability=0.2),
    ])


def create_heavy_augmentation() -> CompositeAugmentation:
    """Create heavy augmentation pipeline."""
    return CompositeAugmentation([
        ObservationNoise(position_std=0.02, velocity_std=0.04, probability=0.7),
        ActionNoise(std=0.02, probability=0.5),
        TemporalShift(max_shift=0.2, probability=0.4),
    ])


class AugmentedDataset:
    """Dataset wrapper that applies augmentation."""
    
    def __init__(self, dataset, augmentation: Optional[CompositeAugmentation] = None, 
                 training: bool = True):
        self.dataset = dataset
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        obs, action = self.dataset[idx]
        
        # Only apply augmentation during training
        if self.training and self.augmentation is not None:
            obs, action = self.augmentation(obs, action)
        
        return obs, action
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def create_augmented_dataloader(dataset, augmentation: Optional[CompositeAugmentation] = None,
                              training: bool = True, **dataloader_kwargs):
    """Create a dataloader with augmentation."""
    augmented_dataset = AugmentedDataset(dataset, augmentation, training)
    
    # Import here to avoid circular imports
    try:
        import torch
        from torch.utils.data import DataLoader
        return DataLoader(augmented_dataset, **dataloader_kwargs)
    except ImportError:
        # Fallback for non-PyTorch environments
        return augmented_dataset


# =============================================================================
# Temporal Data Augmentation
# =============================================================================

class TemporalAugmentation:
    """Base class for temporal sequence augmentation."""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to observation and action sequences."""
        if random.random() < self.probability:
            return self._augment(obs_seq, action_seq)
        return obs_seq, action_seq
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Implement specific temporal augmentation."""
        raise NotImplementedError


class TemporalJittering(TemporalAugmentation):
    """Apply small random time shifts to sequences."""
    
    def __init__(self, max_shift: int = 5, probability: float = 0.4):
        super().__init__(probability)
        self.max_shift = max_shift
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply temporal jittering to sequences."""
        seq_len = len(obs_seq)
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        
        if shift == 0:
            return obs_seq, action_seq
        
        # Create shifted sequences
        jittered_obs = np.zeros_like(obs_seq)
        jittered_action = np.zeros_like(action_seq)
        
        if shift > 0:
            # Shift right (pad with zeros at beginning)
            jittered_obs[shift:] = obs_seq[:-shift]
            jittered_action[shift:] = action_seq[:-shift]
        else:
            # Shift left (pad with zeros at end)
            jittered_obs[:shift] = obs_seq[-shift:]
            jittered_action[:shift] = action_seq[-shift:]
        
        return jittered_obs, jittered_action


class TemporalNoise(TemporalAugmentation):
    """Add temporal noise to sequences."""
    
    def __init__(self, obs_noise_std: float = 0.01, action_noise_std: float = 0.005, 
                 probability: float = 0.4):
        super().__init__(probability)
        self.obs_noise_std = obs_noise_std
        self.action_noise_std = action_noise_std
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add temporal noise to sequences."""
        noisy_obs = obs_seq.copy()
        noisy_action = action_seq.copy()
        
        # Add noise to observations
        obs_noise = np.random.normal(0, self.obs_noise_std, obs_seq.shape)
        noisy_obs += obs_noise
        
        # Add noise to actions
        action_noise = np.random.normal(0, self.action_noise_std, action_seq.shape)
        noisy_action += action_noise
        
        # Clip actions to valid range
        noisy_action = np.clip(noisy_action, -1.0, 1.0)
        
        return noisy_obs, noisy_action


class CompositeTemporalAugmentation:
    """Combine multiple temporal augmentations."""
    
    def __init__(self, augmentations: List[TemporalAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all temporal augmentations in sequence."""
        current_obs, current_action = obs_seq, action_seq
        
        for aug in self.augmentations:
            current_obs, current_action = aug(current_obs, current_action)
        
        return current_obs, current_action


def create_default_temporal_augmentation() -> CompositeTemporalAugmentation:
    """Create default temporal augmentation pipeline."""
    return CompositeTemporalAugmentation([
        TemporalJittering(max_shift=3, probability=0.4),
        TemporalNoise(obs_noise_std=0.005, action_noise_std=0.002, probability=0.4),
    ])


def create_light_temporal_augmentation() -> CompositeTemporalAugmentation:
    """Create light temporal augmentation pipeline."""
    return CompositeTemporalAugmentation([
        TemporalJittering(max_shift=2, probability=0.2),
        TemporalNoise(obs_noise_std=0.002, action_noise_std=0.001, probability=0.2),
    ])


def create_heavy_temporal_augmentation() -> CompositeTemporalAugmentation:
    """Create heavy temporal augmentation pipeline."""
    return CompositeTemporalAugmentation([
        TemporalJittering(max_shift=5, probability=0.6),
        TemporalNoise(obs_noise_std=0.01, action_noise_std=0.005, probability=0.6),
    ])


class AugmentedSequenceDataset:
    """Dataset wrapper that applies temporal augmentation to sequences."""
    
    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]], 
                 augmentation: Optional[CompositeTemporalAugmentation] = None,
                 training: bool = True):
        self.sequences = sequences
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        obs_seq, action_seq = self.sequences[idx]
        
        # Only apply augmentation during training
        if self.training and self.augmentation is not None:
            obs_seq, action_seq = self.augmentation(obs_seq, action_seq)
        
        return obs_seq, action_seq
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# Advanced augmentation techniques have been archived to _archive/bc/data/augmentation_advanced.py

