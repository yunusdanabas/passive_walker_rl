"""
bc/dataset.py

BC dataset loader (scaffold).

Assumes NPZ files from FSM collector with at minimum:
  obs (T+1, 11), act (T, 3) [optional], rew (T,), done (T,)
Optionally: label_act (T, 3) in [-1,1], label_qdes (T,3) in physical units.
"""

from __future__ import annotations
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import Normalizer

REQUIRED = ["obs", "rew", "done"]
OPTIONAL = ["act", "label_act", "label_qdes"]

# Section mappings for action indices
SECTION_MAPPINGS = {
    "hip": [0],           # hip only
    "knees": [1, 2],      # left_knee, right_knee
    "both": [0, 1, 2],    # all three
    "both-adv": [0, 1, 2] # all three (same as both)
}


def discover_npzs(data_dir: str) -> List[str]:
    """Discover all episode NPZ files in the given directory."""
    fs = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not fs:
        raise FileNotFoundError(f"No NPZ episodes found in {data_dir}")
    return fs


def quick_schema_check(path: str) -> Dict[str, Tuple[int, ...]]:
    """Check if NPZ file has required keys and return shapes."""
    with np.load(path) as d:
        shapes = {k: tuple(d[k].shape) for k in d.files}
    for r in REQUIRED:
        if r not in shapes:
            raise ValueError(f"{path} missing required key: {r}")
    return shapes


def split_by_episode(files: List[str], val_ratio: float = 0.1):
    """Split files by episode (avoid data leakage)."""
    n = len(files)
    if n == 1:
        # Special case: use the same file for both train and val
        return files, files
    v = max(1, int(round(n * val_ratio)))
    # Ensure we have at least one training file
    if v >= n:
        v = max(1, n - 1)
    return files[:-v], files[-v:]  # train, val (by index)


def load_xy(files: List[str], section: str, label_type: str = "act") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X (observations) and y (targets) from NPZ files for given section.
    
    Args:
        files: List of NPZ file paths
        section: One of 'hip', 'knees', 'both', 'both-adv'
        label_type: 'act' for normalized actions or 'qdes' for joint positions
        
    Returns:
        X: (N, 11) observations (obs[:-1] per episode)
        y: (N, out_dim) targets (selected action indices)
    """
    if section not in SECTION_MAPPINGS:
        raise ValueError(f"Unknown section: {section}")
    
    action_indices = SECTION_MAPPINGS[section]
    out_dim = len(action_indices)
    
    all_obs = []
    all_targets = []
    
    for file_path in files:
        with np.load(file_path) as data:
            obs = data["obs"]  # (T+1, 11)
            rew = data["rew"]  # (T,)
            done = data["done"]  # (T,)
            
            # Use obs[:-1] as inputs (remove last timestep)
            X_ep = obs[:-1]  # (T, 11)
            
            # Get targets based on label_type
            if label_type == "act" and "label_act" in data:
                targets = data["label_act"]  # (T, 3) in [-1, 1]
            elif label_type == "qdes" and "label_qdes" in data:
                targets = data["label_qdes"]  # (T, 3) in physical units
                # Convert to [-1, 1] using joint ranges (placeholder for now)
                # TODO: Implement proper joint range mapping
                targets = np.clip(targets, -1, 1)
            elif "act" in data:
                # Fallback to regular actions
                targets = data["act"]  # (T, 3)
                targets = np.clip(targets, -1, 1)
            else:
                # TODO: Implement FSM relabeling fallback
                raise ValueError(f"No suitable targets found in {file_path}. "
                               f"Need 'label_act', 'label_qdes', or 'act'.")
            
            # Select relevant action dimensions
            y_ep = targets[:, action_indices]  # (T, out_dim)
            
            all_obs.append(X_ep)
            all_targets.append(y_ep)
    
    X = np.concatenate(all_obs, axis=0)  # (N, 11)
    y = np.concatenate(all_targets, axis=0)  # (N, out_dim)
    
    return X, y


def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int = 1024, 
                      shuffle: bool = True, device: str = "cpu"):
    """
    Create a simple data loader for training.
    
    Args:
        X: Input features (N, 11)
        y: Targets (N, out_dim)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        device: Device to place tensors on
        
    Returns:
        Generator yielding (X_batch, y_batch) tuples
    """
    import torch
    
    N = X.shape[0]
    indices = np.arange(N)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = torch.tensor(X[batch_indices], dtype=torch.float32, device=device)
        y_batch = torch.tensor(y[batch_indices], dtype=torch.float32, device=device)
        
        yield X_batch, y_batch
