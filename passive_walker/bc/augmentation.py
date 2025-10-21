"""
Data Augmentation for BC Training

Online augmentation during training to improve robustness.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any
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


class ScaleAugmentation(BaseAugmentation):
    """Scale observations (simulate different body sizes)."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.95, 1.05), probability: float = 0.3):
        super().__init__(probability)
        self.scale_range = scale_range
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply scaling to observations."""
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        
        augmented_obs = obs.copy()
        
        # Scale position components
        position_indices = [0, 1, 2, 5, 6, 7]
        for i in position_indices:
            if i < len(augmented_obs):
                augmented_obs[i] *= scale_factor
        
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
        ScaleAugmentation(scale_range=(0.95, 1.05), probability=0.3),
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
        ScaleAugmentation(scale_range=(0.9, 1.1), probability=0.5),
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

