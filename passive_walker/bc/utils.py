"""
Small helpers: seeding, device selection, JSON I/O, safe mkdir.
"""
from __future__ import annotations
import os
import json
import random
import numpy as np


def set_seed(seed: int | None):
    """Set random seeds for reproducibility."""
    if seed is None: 
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): 
            torch.cuda.manual_seed_all(seed)
    except Exception: 
        pass
    try:
        import jax
        jax.random.PRNGKey(seed)
    except Exception: 
        pass


def ensure_dir(p: str):
    """Create directory if it doesn't exist."""
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: dict):
    """Save dictionary as JSON file."""
    with open(path, "w") as f: 
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    """Load JSON file as dictionary."""
    with open(path, "r") as f: 
        return json.load(f)


def pick_device(use_gpu: bool) -> str:
    """Select appropriate device (cuda/cpu) based on availability."""
    try:
        import torch
        return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_global_seed(seed: int):
    """Set global random seeds for all backends."""
    set_seed(seed)
    try:
        import jax
        jax.random.PRNGKey(seed)
    except Exception:
        pass


def ckpt_name_for(backend: str, section: str, seed: int, episodes: int, steps: int) -> str:
    """Generate checkpoint filename."""
    return f"{backend}_{section}_seed{seed}_ep{episodes}_steps{steps}"


def meta_name_for(backend: str, section: str, seed: int, episodes: int, steps: int) -> str:
    """Generate metadata filename."""
    return f"{backend}_{section}_seed{seed}_ep{episodes}_steps{steps}_meta.json"


def metrics_name_for(backend: str, section: str, seed: int) -> str:
    """Generate metrics filename."""
    return f"{backend}_{section}_seed{seed}_metrics.json"


def save_metrics_json(filepath: str, train_loss: list, val_loss: list):
    """Save training metrics to JSON."""
    metrics = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epochs": list(range(len(train_loss)))
    }
    save_json(filepath, metrics)


class Normalizer:
    """Simple normalizer for input features."""
    
    def __init__(self, mean=None, std=None, eps=1e-6):
        self.mean = np.asarray(mean) if mean is not None else None
        self.std = np.asarray(std) if std is not None else None
        self.eps = eps
    
    def fit(self, x):
        """Fit normalizer to data."""
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        return self
    
    def apply(self, x):
        """Apply normalization."""
        return self.encode(x)
    
    def encode(self, x):
        """Normalize input features."""
        return (x - self.mean) / (self.std + self.eps)
    
    def decode(self, z):
        """Denormalize features."""
        return z * (self.std + self.eps) + self.mean


def save_checkpoint(model, normalizer, meta, save_dir, section, seed, epoch, steps):
    """Save model checkpoint with metadata."""
    import torch
    import os
    from datetime import datetime
    
    # Create filename
    filename = f"torch_{section}_seed{seed}_ep{epoch}_steps{steps}.pt"
    filepath = os.path.join(save_dir, filename)
    
    # Save model state
    torch.save(model.state_dict(), filepath)
    
    # Save metadata
    meta_file = os.path.join(save_dir, f"torch_{section}_seed{seed}_ep{epoch}_steps{steps}_meta.json")
    meta["checkpoint_file"] = filename
    meta["timestamp"] = datetime.now().isoformat()
    meta["normalizer_mean"] = normalizer.mean.tolist()
    meta["normalizer_std"] = normalizer.std.tolist()
    save_json(meta_file, meta)
    
    return filepath, meta_file


def load_checkpoint(checkpoint_path, model, device="cpu"):
    """Load model checkpoint."""
    import torch
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


class MetricsWriter:
    """Simple metrics tracking for training."""
    
    def __init__(self):
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "epochs": []
        }
    
    def log_epoch(self, epoch, train_loss, val_loss):
        """Log metrics for an epoch."""
        self.metrics["epochs"].append(epoch)
        self.metrics["train_loss"].append(float(train_loss))
        self.metrics["val_loss"].append(float(val_loss))
    
    def save(self, filepath):
        """Save metrics to JSON file."""
        save_json(filepath, self.metrics)
    
    def get_best_epoch(self):
        """Get epoch with lowest validation loss."""
        if not self.metrics["val_loss"]:
            return 0
        return self.metrics["epochs"][np.argmin(self.metrics["val_loss"])]
