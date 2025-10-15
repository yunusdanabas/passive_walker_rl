"""
BC Training Utilities

Helper functions for seeding, device selection, data normalization, 
checkpointing, and metrics tracking.
"""

from __future__ import annotations
import os
import json
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def set_seed(seed: int | None):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed (None for non-deterministic)
    """
    if seed is None: 
        return
    random.seed(seed)
    np.random.seed(seed)
    
    # PyTorch seeding
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
    except Exception: 
        pass
    
    # JAX seeding
    try:
        import jax
        jax.random.PRNGKey(seed)
    except Exception: 
        pass


def set_global_seed(seed: int):
    """Set global seed for JAX (alias for set_seed)."""
    set_seed(seed)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: dict):
    """Save dictionary as JSON file."""
    with open(path, "w") as f: 
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    """Load JSON file as dictionary."""
    with open(path, "r") as f: 
        return json.load(f)


def pick_device(use_gpu: bool) -> str:
    """
    Select appropriate device based on availability and preference.
    
    Args:
        use_gpu: Whether to prefer GPU if available
        
    Returns:
        Device string ("cuda" or "cpu")
    """
    try:
        import torch
        if use_gpu and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Normalizer:
    """
    Data normalization utility for BC training.
    
    Supports both PyTorch-style (encode/decode) and JAX-style (apply) interfaces.
    """
    
    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        """
        Initialize normalizer.
        
        Args:
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.mean = mean
        self.std = std
    
    def fit(self, data: np.ndarray) -> 'Normalizer':
        """
        Fit normalizer to data (JAX-style interface).
        
        Args:
            data: Training data to fit on
            
        Returns:
            Self for chaining
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Avoid division by zero
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self
    
    def encode(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data (PyTorch-style interface).
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return (data - self.mean) / self.std
    
    def decode(self, data: np.ndarray) -> np.ndarray:
        """
        Denormalize data (PyTorch-style interface).
        
        Args:
            data: Normalized data
            
        Returns:
            Denormalized data
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return data * self.std + self.mean
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply normalization (JAX-style interface, alias for encode).
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        return self.encode(data)


class MetricsWriter:
    """Track and save training metrics."""
    
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    
    def save(self, path: str):
        """
        Save metrics to JSON file.
        
        Args:
            path: Output file path
        """
        metrics = {
            "epochs": self.epochs,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses
        }
        save_json(path, metrics)


def ckpt_name_for(backend: str, section: str, seed: int, episodes: int, steps: int) -> str:
    """Generate checkpoint filename."""
    return f"{backend}_{section}_seed{seed}_ep{episodes}_steps{steps}"


def meta_name_for(backend: str, section: str, seed: int, episodes: int, steps: int) -> str:
    """Generate metadata filename."""
    return f"{backend}_{section}_seed{seed}_ep{episodes}_steps{steps}_meta.json"


def metrics_name_for(backend: str, section: str, seed: int) -> str:
    """Generate metrics filename."""
    return f"{backend}_{section}_seed{seed}_metrics.json"


def save_checkpoint(model, normalizer: Normalizer, meta: dict, save_dir: str, 
                   section: str, seed: int, epoch: int, steps: int) -> Tuple[str, str]:
    """
    Save PyTorch model checkpoint and metadata.
    
    Args:
        model: PyTorch model to save
        normalizer: Fitted normalizer
        meta: Metadata dictionary
        save_dir: Directory to save to
        section: Control section
        seed: Random seed
        epoch: Epoch number
        steps: Number of training steps
        
    Returns:
        Tuple of (checkpoint_path, metadata_path)
    """
    import torch
    
    # Generate filenames
    stem = ckpt_name_for("torch", section, seed, 1, steps)  # episodes=1 for single checkpoint
    ckpt_path = os.path.join(save_dir, stem + ".pt")
    meta_path = os.path.join(save_dir, meta_name_for("torch", section, seed, 1, steps))
    
    # Save model
    torch.save(model.state_dict(), ckpt_path)
    
    # Add normalizer info to metadata
    meta["normalizer_mean"] = normalizer.mean.tolist()
    meta["normalizer_std"] = normalizer.std.tolist()
    
    # Save metadata
    save_json(meta_path, meta)
    
    return ckpt_path, meta_path


def load_checkpoint(ckpt_path: str, model) -> 'torch.nn.Module':
    """
    Load PyTorch model checkpoint.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        
    Returns:
        Model with loaded weights
    """
    import torch
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model


def save_metrics_json(path: str, train_losses: List[float], val_losses: List[float]):
    """
    Save training metrics to JSON file.
    
    Args:
        path: Output file path
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs": list(range(len(train_losses)))
    }
    save_json(path, metrics)


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def format_size(bytes_size: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}TB"