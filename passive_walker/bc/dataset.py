"""
BC Dataset Loading and Processing

Handles loading and preprocessing of FSM demonstration data for BC training.
Supports different control sections and frame stacking for temporal context.
"""

from __future__ import annotations
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import Normalizer

# Required keys in NPZ files
REQUIRED = ["obs", "rew", "done"]
OPTIONAL = ["act", "label_act", "label_qdes", "info_qdes"]

# Section mappings for action indices
SECTION_MAPPINGS = {
    "hip": [0],           # hip only
    "knees": [1, 2],      # left_knee, right_knee
    "both": [0, 1, 2],    # all three joints
    "both-adv": [0, 1, 2] # all three joints (same as both)
}


def discover_npzs(data_dir: str) -> List[str]:
    """
    Discover all episode NPZ files in the given directory.
    
    Args:
        data_dir: Directory containing episode_*.npz files
        
    Returns:
        Sorted list of NPZ file paths
        
    Raises:
        FileNotFoundError: If no NPZ files found
    """
    fs = sorted(glob.glob(os.path.join(data_dir, "episode_*.npz")))
    if not fs:
        raise FileNotFoundError(f"No NPZ episodes found in {data_dir}")
    return fs


def quick_schema_check(path: str) -> Dict[str, Tuple[int, ...]]:
    """
    Check if NPZ file has required keys and return shapes.
    
    Args:
        path: Path to NPZ file
        
    Returns:
        Dictionary mapping keys to their shapes
        
    Raises:
        ValueError: If required keys are missing
    """
    with np.load(path) as d:
        shapes = {k: tuple(d[k].shape) for k in d.files}
    for r in REQUIRED:
        if r not in shapes:
            raise ValueError(f"{path} missing required key: {r}")
    return shapes


def split_by_episode(files: List[str], val_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    """
    Split files by episode to avoid data leakage.
    
    Args:
        files: List of NPZ file paths
        val_ratio: Fraction of files to use for validation
        
    Returns:
        Tuple of (train_files, val_files)
    """
    n = len(files)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    
    # Use last episodes for validation (chronological split)
    train_files = files[:n_train]
    val_files = files[n_train:]
    
    return train_files, val_files


def load_xy(files: List[str], section: str, label_type: str = "act", frame_stack: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process data for BC training.
    
    Args:
        files: List of NPZ file paths
        section: Control section ("hip", "knees", "both", "both-adv")
        label_type: Label type ("act" for actions, "qdes" for desired joint positions)
        frame_stack: Number of frames to stack for temporal context
        
    Returns:
        Tuple of (X, y) arrays for training
        
    Raises:
        ValueError: If section or label_type is invalid
    """
    if section not in SECTION_MAPPINGS:
        raise ValueError(f"Unknown section: {section}. Must be one of {list(SECTION_MAPPINGS.keys())}")
    
    if label_type not in ["act", "qdes"]:
        raise ValueError(f"Unknown label_type: {label_type}. Must be 'act' or 'qdes'")
    
    # Collect all data
    all_obs = []
    all_labels = []
    
    for file_path in files:
        with np.load(file_path) as data:
            obs = data["obs"].astype(np.float32)  # (T+1, 11)
            rew = data["rew"].astype(np.float32)  # (T,)
            done = data["done"].astype(bool)      # (T,)
            
            # Get labels based on type
            if label_type == "act":
                if "act" in data:
                    labels = data["act"].astype(np.float32)  # (T, 3)
                else:
                    raise ValueError(f"No 'act' data in {file_path}")
            else:  # qdes
                if "info_qdes" in data:
                    labels = data["info_qdes"].astype(np.float32)  # (T, 3)
                else:
                    raise ValueError(f"No 'info_qdes' data in {file_path}")
            
            # Extract relevant joint indices
            joint_indices = SECTION_MAPPINGS[section]
            labels = labels[:, joint_indices]  # (T, n_joints)
            
            # Process episode
            episode_obs, episode_labels = _process_episode(obs, labels, rew, done, frame_stack)
            
            if len(episode_obs) > 0:
                all_obs.append(episode_obs)
                all_labels.append(episode_labels)
    
    if not all_obs:
        raise ValueError("No valid episodes found")
    
    # Concatenate all episodes
    X = np.concatenate(all_obs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y


def _process_episode(obs: np.ndarray, labels: np.ndarray, rew: np.ndarray, done: np.ndarray, frame_stack: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single episode for BC training.
    
    Args:
        obs: Observations (T+1, 11)
        labels: Labels (T, n_joints)
        rew: Rewards (T,)
        done: Done flags (T,)
        frame_stack: Number of frames to stack
        
    Returns:
        Tuple of (processed_obs, processed_labels)
    """
    T = len(labels)
    
    if frame_stack == 1:
        # Simple case: no frame stacking
        X = obs[:-1]  # Remove last observation (T, 11)
        y = labels    # (T, n_joints)
    else:
        # Frame stacking: create temporal windows
        X = []
        y = []
        
        for t in range(T):
            # Check if we have enough history
            if t < frame_stack - 1:
                continue
                
            # Stack frames
            start_idx = t - frame_stack + 1
            end_idx = t + 1
            stacked_obs = obs[start_idx:end_idx].flatten()  # (frame_stack * 11,)
            
            X.append(stacked_obs)
            y.append(labels[t])
        
        X = np.array(X) if X else np.empty((0, frame_stack * 11), dtype=np.float32)
        y = np.array(y) if y else np.empty((0, labels.shape[1]), dtype=np.float32)
    
    return X, y


def create_data_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str = "cpu"):
    """
    Create PyTorch data loader for training.
    
    Args:
        X: Input features (N, input_dim)
        y: Target labels (N, output_dim)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        device: Device to use ("cpu" or "cuda")
        
    Yields:
        Batches of (X_batch, y_batch) as tensors
    """
    import torch
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=False
    )
    
    return dataloader


def validate_dataset(files: List[str], section: str, label_type: str = "act") -> Dict[str, any]:
    """
    Validate dataset and return statistics.
    
    Args:
        files: List of NPZ file paths
        section: Control section
        label_type: Label type
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        "n_files": len(files),
        "n_episodes": 0,
        "total_steps": 0,
        "avg_episode_length": 0.0,
        "input_dim": 0,
        "output_dim": 0,
        "section": section,
        "label_type": label_type
    }
    
    if not files:
        return stats
    
    # Load a sample to get dimensions
    try:
        X, y = load_xy(files[:1], section, label_type, frame_stack=1)
        stats["input_dim"] = X.shape[1]
        stats["output_dim"] = y.shape[1]
    except Exception as e:
        stats["error"] = str(e)
        return stats
    
    # Count total steps across all files
    total_steps = 0
    for file_path in files:
        with np.load(file_path) as data:
            T = len(data["rew"])
            total_steps += T
    
    stats["total_steps"] = total_steps
    stats["n_episodes"] = len(files)
    stats["avg_episode_length"] = total_steps / len(files) if files else 0.0
    
    return stats