"""
Distributed Training Support

Multi-GPU training utilities and parallel data loading.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any, Optional, List
import numpy as np


class DistributedTrainingManager:
    """
    Simple distributed training manager.
    
    Handles DDP setup, data loading, and synchronization.
    """
    
    def __init__(self, backend: str = "nccl"):
        """
        Initialize distributed training manager.
        
        Args:
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        """
        self.backend = backend
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_distributed = False
        
        # Initialize if environment variables are set
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._init_distributed()
    
    def _init_distributed(self):
        """Initialize distributed training."""
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize process group
        dist.init_process_group(backend=self.backend)
        self.is_distributed = True
        
        print(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
    
    def wrap_model(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        Wrap model with DDP if distributed training is enabled.
        
        Args:
            model: PyTorch model
            device: Device to place model on
            
        Returns:
            Wrapped model
        """
        model = model.to(device)
        
        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank])
        
        return model
    
    def create_dataloader(self, dataset, batch_size: int, 
                         shuffle: bool = True, num_workers: int = 4,
                         pin_memory: bool = True) -> DataLoader:
        """
        Create distributed data loader.
        
        Args:
            dataset: Dataset to load
            batch_size: Batch size per GPU
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            Data loader
        """
        sampler = None
        
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # Sampler handles shuffling
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def synchronize(self):
        """Synchronize all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def cleanup(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()


class MixedPrecisionManager:
    """
    Simple mixed precision training manager.
    
    Handles automatic mixed precision (AMP) for memory efficiency.
    """
    
    def __init__(self, enabled: bool = True, loss_scale: str = "dynamic"):
        """
        Initialize mixed precision manager.
        
        Args:
            enabled: Whether to enable mixed precision
            loss_scale: Loss scaling strategy ("dynamic" or "static")
        """
        self.enabled = enabled
        self.loss_scale = loss_scale
        
        if enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def autocast(self):
        """Get autocast context manager."""
        if self.enabled:
            return torch.cuda.amp.autocast()
        else:
            return torch.cuda.amp.autocast(enabled=False)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss for mixed precision.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer: torch.optim.Optimizer):
        """
        Step optimizer with mixed precision.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def update(self):
        """Update scaler."""
        if self.scaler is not None:
            self.scaler.update()


class GradientAccumulator:
    """
    Simple gradient accumulation manager.
    
    Accumulates gradients over multiple steps for large effective batch sizes.
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Increment step counter."""
        self.current_step += 1
    
    def reset(self):
        """Reset step counter."""
        self.current_step = 0


class MemoryProfiler:
    """
    Simple memory profiler for training.
    
    Tracks GPU memory usage and provides optimization suggestions.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            enabled: Whether to enable profiling
        """
        self.enabled = enabled
        self.memory_history = []
    
    def log_memory_usage(self, step: int, phase: str = "train"):
        """
        Log current memory usage.
        
        Args:
            step: Current step
            phase: Training phase ("train", "val", "test")
        """
        if not self.enabled or not torch.cuda.is_available():
            return
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        
        self.memory_history.append({
            "step": step,
            "phase": phase,
            "allocated": memory_allocated,
            "reserved": memory_reserved
        })
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary of memory stats
        """
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved": torch.cuda.max_memory_reserved() / 1024**3
        }
    
    def clear_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_optimization_suggestions(self) -> List[str]:
        """
        Get memory optimization suggestions.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if not torch.cuda.is_available():
            return suggestions
        
        current_stats = self.get_memory_stats()
        
        if current_stats["allocated"] > 8.0:  # More than 8GB
            suggestions.append("Consider reducing batch size")
            suggestions.append("Enable mixed precision training")
        
        if current_stats["reserved"] > current_stats["allocated"] * 1.5:
            suggestions.append("Clear GPU cache with torch.cuda.empty_cache()")
        
        if len(self.memory_history) > 10:
            recent_memory = [h["allocated"] for h in self.memory_history[-10:]]
            if max(recent_memory) - min(recent_memory) > 2.0:
                suggestions.append("Memory usage is fluctuating - check for memory leaks")
        
        return suggestions


def setup_distributed_training(backend: str = "nccl") -> DistributedTrainingManager:
    """
    Setup distributed training.
    
    Args:
        backend: Distributed backend
        
    Returns:
        Distributed training manager
    """
    return DistributedTrainingManager(backend)


def setup_mixed_precision(enabled: bool = True) -> MixedPrecisionManager:
    """
    Setup mixed precision training.
    
    Args:
        enabled: Whether to enable mixed precision
        
    Returns:
        Mixed precision manager
    """
    return MixedPrecisionManager(enabled)


def setup_memory_profiler(enabled: bool = True) -> MemoryProfiler:
    """
    Setup memory profiler.
    
    Args:
        enabled: Whether to enable profiling
        
    Returns:
        Memory profiler
    """
    return MemoryProfiler(enabled)


def create_distributed_dataloader(dataset, batch_size: int, 
                                 distributed_manager: DistributedTrainingManager,
                                 **kwargs) -> DataLoader:
    """
    Create distributed data loader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size per GPU
        distributed_manager: Distributed training manager
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Distributed data loader
    """
    return distributed_manager.create_dataloader(dataset, batch_size, **kwargs)
