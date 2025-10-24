"""
Data Augmentation for BC Training

Online augmentation during training to improve robustness.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import random
from scipy import interpolate


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


class TimeWarping(TemporalAugmentation):
    """Apply time warping to sequences (speed up/slow down)."""
    
    def __init__(self, warp_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.3):
        super().__init__(probability)
        self.warp_range = warp_range
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply time warping to sequences."""
        seq_len = len(obs_seq)
        warp_factor = np.random.uniform(self.warp_range[0], self.warp_range[1])
        
        # Create new time indices
        original_indices = np.arange(seq_len)
        new_indices = original_indices * warp_factor
        
        # Interpolate sequences
        warped_obs = np.zeros_like(obs_seq)
        warped_action = np.zeros_like(action_seq)
        
        for i in range(obs_seq.shape[1]):
            f_obs = interpolate.interp1d(original_indices, obs_seq[:, i], 
                                       kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_obs[:, i] = f_obs(new_indices)
        
        for i in range(action_seq.shape[1]):
            f_action = interpolate.interp1d(original_indices, action_seq[:, i], 
                                          kind='linear', bounds_error=False, fill_value='extrapolate')
            warped_action[:, i] = f_action(new_indices)
        
        return warped_obs, warped_action


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


class SubsequenceExtraction(TemporalAugmentation):
    """Extract random subsequences from episodes."""
    
    def __init__(self, min_length_ratio: float = 0.5, max_length_ratio: float = 1.0, 
                 probability: float = 0.3):
        super().__init__(probability)
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random subsequence."""
        seq_len = len(obs_seq)
        min_length = max(10, int(seq_len * self.min_length_ratio))
        max_length = min(seq_len, int(seq_len * self.max_length_ratio))
        
        subseq_length = np.random.randint(min_length, max_length + 1)
        start_idx = np.random.randint(0, seq_len - subseq_length + 1)
        end_idx = start_idx + subseq_length
        
        return obs_seq[start_idx:end_idx], action_seq[start_idx:end_idx]


class FrameDropout(TemporalAugmentation):
    """Randomly drop frames from sequences (robust to missing data)."""
    
    def __init__(self, dropout_rate: float = 0.1, probability: float = 0.2):
        super().__init__(probability)
        self.dropout_rate = dropout_rate
    
    def _augment(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply frame dropout."""
        seq_len = len(obs_seq)
        n_drop = int(seq_len * self.dropout_rate)
        
        if n_drop == 0:
            return obs_seq, action_seq
        
        # Randomly select frames to drop
        drop_indices = np.random.choice(seq_len, size=n_drop, replace=False)
        
        # Create mask
        mask = np.ones(seq_len, dtype=bool)
        mask[drop_indices] = False
        
        # Apply mask
        dropped_obs = obs_seq[mask]
        dropped_action = action_seq[mask]
        
        return dropped_obs, dropped_action


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
        TimeWarping(warp_range=(0.9, 1.1), probability=0.3),
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
        TimeWarping(warp_range=(0.8, 1.2), probability=0.5),
        TemporalJittering(max_shift=5, probability=0.6),
        SubsequenceExtraction(min_length_ratio=0.6, max_length_ratio=0.9, probability=0.3),
        FrameDropout(dropout_rate=0.15, probability=0.3),
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


# =============================================================================
# Advanced Augmentation Techniques
# =============================================================================

class ContactPerturbation(BaseAugmentation):
    """Perturb contact information for robustness."""
    
    def __init__(self, contact_noise_std: float = 0.1, force_noise_std: float = 0.05, 
                 probability: float = 0.3):
        super().__init__(probability)
        self.contact_noise_std = contact_noise_std
        self.force_noise_std = force_noise_std
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perturb contact information."""
        augmented_obs = obs.copy()
        
        # Contact flags (indices 11, 12) - add noise and threshold
        if len(obs) > 12:
            contact_noise = np.random.normal(0, self.contact_noise_std, 2)
            augmented_obs[11] = np.clip(obs[11] + contact_noise[0], 0, 1)
            augmented_obs[12] = np.clip(obs[12] + contact_noise[1], 0, 1)
        
        # Contact forces (indices 13, 14) - add noise
        if len(obs) > 14:
            force_noise = np.random.normal(0, self.force_noise_std, 2)
            augmented_obs[13] += force_noise[0]
            augmented_obs[14] += force_noise[1]
        
        return augmented_obs, action


class AdaptiveNoise(BaseAugmentation):
    """Adaptive noise based on observation magnitude."""
    
    def __init__(self, base_std: float = 0.01, adaptive_factor: float = 0.1, 
                 probability: float = 0.4):
        super().__init__(probability)
        self.base_std = base_std
        self.adaptive_factor = adaptive_factor
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adaptive noise."""
        augmented_obs = obs.copy()
        
        # Compute adaptive noise based on observation magnitude
        obs_magnitude = np.abs(obs)
        adaptive_std = self.base_std + self.adaptive_factor * obs_magnitude
        
        # Add noise
        noise = np.random.normal(0, adaptive_std)
        augmented_obs += noise
        
        return augmented_obs, action


class AdversarialPerturbation(BaseAugmentation):
    """Simple adversarial perturbation (FGSM-like)."""
    
    def __init__(self, epsilon: float = 0.01, probability: float = 0.2):
        super().__init__(probability)
        self.epsilon = epsilon
    
    def _augment(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adversarial perturbation."""
        # Simple implementation: add epsilon * sign of gradient approximation
        # In practice, this would use the actual gradient from the model
        perturbation = self.epsilon * np.sign(np.random.randn(*obs.shape))
        augmented_obs = obs + perturbation
        
        return augmented_obs, action


class AdaptiveAugmentation:
    """Adaptive augmentation based on validation performance."""
    
    def __init__(self, base_augmentation: CompositeAugmentation, 
                 adaptation_rate: float = 0.1, min_probability: float = 0.1):
        self.base_augmentation = base_augmentation
        self.adaptation_rate = adaptation_rate
        self.min_probability = min_probability
        self.validation_losses = []
        self.augmentation_strength = 1.0
    
    def update_validation_loss(self, val_loss: float):
        """Update validation loss and adjust augmentation strength."""
        self.validation_losses.append(val_loss)
        
        if len(self.validation_losses) >= 2:
            # If validation loss is increasing, reduce augmentation
            if self.validation_losses[-1] > self.validation_losses[-2]:
                self.augmentation_strength *= (1 - self.adaptation_rate)
            else:
                # If validation loss is decreasing, increase augmentation
                self.augmentation_strength *= (1 + self.adaptation_rate)
            
            # Clamp augmentation strength
            self.augmentation_strength = np.clip(
                self.augmentation_strength, 
                self.min_probability, 
                2.0
            )
    
    def __call__(self, obs: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adaptive augmentation."""
        # Temporarily adjust probabilities based on strength
        original_probs = []
        for aug in self.base_augmentation.augmentations:
            original_probs.append(aug.probability)
            aug.probability *= self.augmentation_strength
        
        # Apply augmentation
        result = self.base_augmentation(obs, action)
        
        # Restore original probabilities
        for aug, orig_prob in zip(self.base_augmentation.augmentations, original_probs):
            aug.probability = orig_prob
        
        return result


def create_advanced_augmentation() -> CompositeAugmentation:
    """Create advanced augmentation pipeline."""
    return CompositeAugmentation([
        ObservationNoise(position_std=0.01, velocity_std=0.02, probability=0.5),
        ActionNoise(std=0.01, probability=0.3),
        ContactPerturbation(contact_noise_std=0.1, force_noise_std=0.05, probability=0.3),
        AdaptiveNoise(base_std=0.01, adaptive_factor=0.1, probability=0.4),
        TemporalShift(max_shift=0.1, probability=0.2),
        ScaleAugmentation(scale_range=(0.95, 1.05), probability=0.3),
    ])


def create_robust_augmentation() -> CompositeAugmentation:
    """Create robust augmentation pipeline for domain generalization."""
    return CompositeAugmentation([
        ObservationNoise(position_std=0.02, velocity_std=0.04, probability=0.6),
        ActionNoise(std=0.02, probability=0.4),
        ContactPerturbation(contact_noise_std=0.2, force_noise_std=0.1, probability=0.4),
        AdaptiveNoise(base_std=0.02, adaptive_factor=0.2, probability=0.5),
        AdversarialPerturbation(epsilon=0.01, probability=0.2),
        TemporalShift(max_shift=0.2, probability=0.3),
        ScaleAugmentation(scale_range=(0.9, 1.1), probability=0.4),
    ])


def create_adaptive_temporal_augmentation() -> CompositeTemporalAugmentation:
    """Create adaptive temporal augmentation pipeline."""
    return CompositeTemporalAugmentation([
        TimeWarping(warp_range=(0.8, 1.2), probability=0.4),
        TemporalJittering(max_shift=5, probability=0.5),
        SubsequenceExtraction(min_length_ratio=0.5, max_length_ratio=1.0, probability=0.3),
        FrameDropout(dropout_rate=0.1, probability=0.3),
        TemporalNoise(obs_noise_std=0.01, action_noise_std=0.005, probability=0.5),
    ])

