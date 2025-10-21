"""
Learning Rate Schedulers for BC Training

Supports both PyTorch and JAX backends with various scheduling strategies.
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np


class BaseScheduler:
    """Base class for learning rate schedulers."""
    
    def __init__(self, initial_lr: float, **kwargs):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> float:
        """Update learning rate and return current value."""
        raise NotImplementedError
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class NoScheduler(BaseScheduler):
    """No learning rate scheduling (constant LR)."""
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> float:
        return self.current_lr


class PlateauScheduler(BaseScheduler):
    """Reduce LR on plateau (PyTorch-style)."""
    
    def __init__(self, initial_lr: float, mode: str = 'min', factor: float = 0.5, 
                 patience: int = 10, threshold: float = 1e-4, min_lr: float = 1e-6):
        super().__init__(initial_lr)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        
        self.best_metric = None
        self.patience_counter = 0
        self.num_bad_epochs = 0
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> float:
        if metrics is None:
            return self.current_lr
        
        # Use validation loss as metric
        metric = metrics.get('val_loss', metrics.get('loss', 0.0))
        
        if self.best_metric is None:
            self.best_metric = metric
        elif self._is_better(metric, self.best_metric):
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
        
        return self.current_lr
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'min':
            return current < best - self.threshold
        else:  # max
            return current > best + self.threshold
    
    def _reduce_lr(self):
        """Reduce learning rate."""
        old_lr = self.current_lr
        self.current_lr = max(old_lr * self.factor, self.min_lr)
        print(f"Reducing learning rate from {old_lr:.2e} to {self.current_lr:.2e}")


class CosineScheduler(BaseScheduler):
    """Cosine annealing scheduler."""
    
    def __init__(self, initial_lr: float, T_max: int, eta_min: float = 0.0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> float:
        if epoch >= self.T_max:
            self.current_lr = self.eta_min
        else:
            self.current_lr = (self.eta_min + 
                              (self.initial_lr - self.eta_min) * 
                              (1 + np.cos(np.pi * epoch / self.T_max)) / 2)
        return self.current_lr


class WarmupCosineScheduler(BaseScheduler):
    """Cosine annealing with warmup."""
    
    def __init__(self, initial_lr: float, T_max: int, warmup_epochs: int = 10, eta_min: float = 0.0):
        super().__init__(initial_lr)
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
    
    def step(self, epoch: int, metrics: Optional[Dict[str, float]] = None) -> float:
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr * epoch / self.warmup_epochs
        elif epoch >= self.T_max:
            self.current_lr = self.eta_min
        else:
            # Cosine annealing after warmup
            effective_epoch = epoch - self.warmup_epochs
            effective_T_max = self.T_max - self.warmup_epochs
            self.current_lr = (self.eta_min + 
                              (self.initial_lr - self.eta_min) * 
                              (1 + np.cos(np.pi * effective_epoch / effective_T_max)) / 2)
        return self.current_lr


def create_scheduler(scheduler_type: str, initial_lr: float, **kwargs) -> BaseScheduler:
    """Factory function to create schedulers."""
    if scheduler_type == "none":
        return NoScheduler(initial_lr)
    elif scheduler_type == "plateau":
        return PlateauScheduler(initial_lr, **kwargs)
    elif scheduler_type == "cosine":
        return CosineScheduler(initial_lr, **kwargs)
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(initial_lr, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# JAX-specific schedulers using optax
try:
    import optax
    
    def create_jax_scheduler(scheduler_type: str, initial_lr: float, **kwargs):
        """Create JAX schedulers using optax."""
        if scheduler_type == "none":
            return optax.constant_schedule(initial_lr)
        elif scheduler_type == "cosine":
            T_max = kwargs.get('T_max', 100)
            return optax.cosine_decay_schedule(initial_lr, T_max)
        elif scheduler_type == "warmup_cosine":
            T_max = kwargs.get('T_max', 100)
            warmup_epochs = kwargs.get('warmup_epochs', 10)
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=initial_lr,
                warmup_steps=warmup_epochs,
                decay_steps=T_max - warmup_epochs
            )
        else:
            raise ValueError(f"JAX scheduler not implemented for: {scheduler_type}")
    
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    
    def create_jax_scheduler(scheduler_type: str, initial_lr: float, **kwargs):
        """Fallback when optax is not available."""
        raise ImportError("optax not available. Install with: pip install optax")


# PyTorch-specific schedulers
try:
    import torch
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
    
    def create_torch_scheduler(scheduler_type: str, optimizer, initial_lr: float, **kwargs):
        """Create PyTorch schedulers."""
        if scheduler_type == "none":
            return None
        elif scheduler_type == "plateau":
            return ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                threshold=kwargs.get('threshold', 1e-4),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        elif scheduler_type == "cosine":
            T_max = kwargs.get('T_max', 100)
            return CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_type == "warmup_cosine":
            T_max = kwargs.get('T_max', 100)
            return CosineAnnealingWarmRestarts(optimizer, T_0=T_max)
        else:
            raise ValueError(f"PyTorch scheduler not implemented for: {scheduler_type}")
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
    def create_torch_scheduler(scheduler_type: str, optimizer, initial_lr: float, **kwargs):
        """Fallback when PyTorch is not available."""
        raise ImportError("PyTorch not available. Install with: pip install torch")

