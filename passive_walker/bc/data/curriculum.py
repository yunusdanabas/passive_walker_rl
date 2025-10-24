"""
Curriculum Learning for Behavior Cloning

Progressive difficulty scheduling and adaptive loss weighting.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod


class CurriculumScheduler(ABC):
    """Base class for curriculum learning schedulers."""
    
    @abstractmethod
    def get_current_stage(self, epoch: int) -> Dict[str, Any]:
        """Get current curriculum stage parameters."""
        pass
    
    @abstractmethod
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update curriculum based on training metrics."""
        pass


class ProgressiveDifficultyScheduler(CurriculumScheduler):
    """Progressive difficulty curriculum scheduler."""
    
    def __init__(self, 
                 total_epochs: int,
                 clean_data_ratio: float = 0.8,
                 perturbed_data_ratio: float = 0.2,
                 transition_epochs: int = 20):
        """
        Initialize progressive difficulty scheduler.
        
        Args:
            total_epochs: Total training epochs
            clean_data_ratio: Initial ratio of clean data
            perturbed_data_ratio: Final ratio of perturbed data
            transition_epochs: Epochs for transition
        """
        self.total_epochs = total_epochs
        self.clean_data_ratio = clean_data_ratio
        self.perturbed_data_ratio = perturbed_data_ratio
        self.transition_epochs = transition_epochs
        self.current_stage = 0
    
    def get_current_stage(self, epoch: int) -> Dict[str, Any]:
        """Get current curriculum stage."""
        if epoch < self.transition_epochs:
            # Transition phase
            progress = epoch / self.transition_epochs
            clean_ratio = self.clean_data_ratio - progress * (self.clean_data_ratio - (1 - self.perturbed_data_ratio))
            perturbed_ratio = self.perturbed_data_ratio * progress
        else:
            # Final phase
            clean_ratio = 1 - self.perturbed_data_ratio
            perturbed_ratio = self.perturbed_data_ratio
        
        return {
            "clean_data_ratio": clean_ratio,
            "perturbed_data_ratio": perturbed_ratio,
            "stage": "transition" if epoch < self.transition_epochs else "final",
            "progress": min(epoch / self.transition_epochs, 1.0)
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update curriculum (no adaptive updates for this scheduler)."""
        pass


class SequenceLengthCurriculum(CurriculumScheduler):
    """Sequence length curriculum scheduler."""
    
    def __init__(self,
                 total_epochs: int,
                 min_length: int = 10,
                 max_length: int = 100,
                 transition_epochs: int = 30):
        """
        Initialize sequence length curriculum.
        
        Args:
            total_epochs: Total training epochs
            min_length: Starting sequence length
            max_length: Final sequence length
            transition_epochs: Epochs for transition
        """
        self.total_epochs = total_epochs
        self.min_length = min_length
        self.max_length = max_length
        self.transition_epochs = transition_epochs
    
    def get_current_stage(self, epoch: int) -> Dict[str, Any]:
        """Get current sequence length."""
        if epoch < self.transition_epochs:
            progress = epoch / self.transition_epochs
            current_length = int(self.min_length + progress * (self.max_length - self.min_length))
        else:
            current_length = self.max_length
        
        return {
            "sequence_length": current_length,
            "stage": "transition" if epoch < self.transition_epochs else "final",
            "progress": min(epoch / self.transition_epochs, 1.0)
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update curriculum (no adaptive updates for this scheduler)."""
        pass


class AdaptiveLossWeighting(CurriculumScheduler):
    """Adaptive loss weighting based on performance."""
    
    def __init__(self,
                 base_weights: Dict[str, float],
                 adaptation_rate: float = 0.1,
                 min_weight: float = 0.01,
                 max_weight: float = 2.0):
        """
        Initialize adaptive loss weighting.
        
        Args:
            base_weights: Base loss weights
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum weight value
            max_weight: Maximum weight value
        """
        self.base_weights = base_weights.copy()
        self.current_weights = base_weights.copy()
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.previous_losses = {}
    
    def get_current_stage(self, epoch: int) -> Dict[str, Any]:
        """Get current loss weights."""
        return {
            "loss_weights": self.current_weights.copy(),
            "stage": "adaptive",
            "epoch": epoch
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update loss weights based on metrics."""
        for loss_name, current_loss in metrics.items():
            if loss_name in self.current_weights:
                if loss_name in self.previous_losses:
                    # Compare with previous loss
                    if current_loss > self.previous_losses[loss_name]:
                        # Loss increased, reduce weight
                        self.current_weights[loss_name] *= (1 - self.adaptation_rate)
                    else:
                        # Loss decreased, increase weight
                        self.current_weights[loss_name] *= (1 + self.adaptation_rate)
                    
                    # Clamp weights
                    self.current_weights[loss_name] = np.clip(
                        self.current_weights[loss_name],
                        self.min_weight,
                        self.max_weight
                    )
                
                self.previous_losses[loss_name] = current_loss


class CompositeCurriculum(CurriculumScheduler):
    """Composite curriculum combining multiple schedulers."""
    
    def __init__(self, schedulers: List[CurriculumScheduler]):
        """
        Initialize composite curriculum.
        
        Args:
            schedulers: List of curriculum schedulers
        """
        self.schedulers = schedulers
    
    def get_current_stage(self, epoch: int) -> Dict[str, Any]:
        """Get combined curriculum stage."""
        stage_info = {"epoch": epoch}
        
        for i, scheduler in enumerate(self.schedulers):
            scheduler_stage = scheduler.get_current_stage(epoch)
            stage_info[f"scheduler_{i}"] = scheduler_stage
        
        return stage_info
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update all schedulers."""
        for scheduler in self.schedulers:
            scheduler.update(epoch, metrics)


class CurriculumDataSampler:
    """Data sampler that implements curriculum learning."""
    
    def __init__(self, 
                 clean_data: List[Tuple[np.ndarray, np.ndarray]],
                 perturbed_data: List[Tuple[np.ndarray, np.ndarray]],
                 scheduler: CurriculumScheduler):
        """
        Initialize curriculum data sampler.
        
        Args:
            clean_data: Clean demonstration data
            perturbed_data: Perturbed demonstration data
            scheduler: Curriculum scheduler
        """
        self.clean_data = clean_data
        self.perturbed_data = perturbed_data
        self.scheduler = scheduler
        self.current_epoch = 0
    
    def sample_batch(self, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Sample batch according to current curriculum stage."""
        stage_info = self.scheduler.get_current_stage(self.current_epoch)
        
        if "clean_data_ratio" in stage_info:
            # Progressive difficulty curriculum
            clean_ratio = stage_info["clean_data_ratio"]
            n_clean = int(batch_size * clean_ratio)
            n_perturbed = batch_size - n_clean
            
            # Sample from clean data
            clean_indices = np.random.choice(len(self.clean_data), n_clean, replace=True)
            clean_samples = [self.clean_data[i] for i in clean_indices]
            
            # Sample from perturbed data
            perturbed_indices = np.random.choice(len(self.perturbed_data), n_perturbed, replace=True)
            perturbed_samples = [self.perturbed_data[i] for i in perturbed_indices]
            
            # Combine and shuffle
            all_samples = clean_samples + perturbed_samples
            np.random.shuffle(all_samples)
            
            return all_samples
        else:
            # Default: sample from all data
            all_data = self.clean_data + self.perturbed_data
            indices = np.random.choice(len(all_data), batch_size, replace=True)
            return [all_data[i] for i in indices]
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Update curriculum for new epoch."""
        self.current_epoch = epoch
        self.scheduler.update(epoch, metrics)


def create_default_curriculum(total_epochs: int) -> CompositeCurriculum:
    """Create default curriculum learning setup."""
    schedulers = [
        ProgressiveDifficultyScheduler(total_epochs),
        SequenceLengthCurriculum(total_epochs),
        AdaptiveLossWeighting({
            "l1_loss": 1.0,
            "smoothness_loss": 0.1,
            "bound_penalty": 0.01
        })
    ]
    return CompositeCurriculum(schedulers)


def create_advanced_curriculum(total_epochs: int) -> CompositeCurriculum:
    """Create advanced curriculum learning setup."""
    schedulers = [
        ProgressiveDifficultyScheduler(
            total_epochs,
            clean_data_ratio=0.9,
            perturbed_data_ratio=0.3,
            transition_epochs=25
        ),
        SequenceLengthCurriculum(
            total_epochs,
            min_length=20,
            max_length=200,
            transition_epochs=40
        ),
        AdaptiveLossWeighting({
            "l1_loss": 1.0,
            "mse_loss": 0.5,
            "smoothness_loss": 0.2,
            "bound_penalty": 0.05,
            "temporal_consistency": 0.1
        }, adaptation_rate=0.15)
    ]
    return CompositeCurriculum(schedulers)


def apply_curriculum_loss_weights(losses: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Apply curriculum loss weights to individual losses.
    
    Args:
        losses: Dictionary of loss values
        weights: Dictionary of loss weights
        
    Returns:
        Weighted total loss
    """
    total_loss = 0.0
    
    for loss_name, loss_value in losses.items():
        weight = weights.get(loss_name, 1.0)
        total_loss += weight * loss_value
    
    return total_loss
