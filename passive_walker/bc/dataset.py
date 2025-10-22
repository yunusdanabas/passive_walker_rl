"""
BC Dataset Loading and Processing

Handles loading and preprocessing of FSM demonstration data for BC training.
Supports different control sections and frame stacking.
"""

from __future__ import annotations
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from .utils import Normalizer

# Joint ranges for normalization (matching controller.py)
JOINT_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
JOINT_MAX = np.array([+0.5, +0.5, +0.5], dtype=np.float32)

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
                    # Normalize qdes to [-1, 1] range for each joint
                    for i in range(labels.shape[1]):
                        lo, hi = JOINT_MIN[i], JOINT_MAX[i]
                        # Normalize: (value - lo) / (hi - lo) * 2 - 1 --> [-1, 1]
                        labels[:, i] = 2.0 * (labels[:, i] - lo) / (hi - lo) - 1.0
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


# =============================================================================
# Temporal Sequence Loading Functions
# =============================================================================

def load_sequences(files: List[str], section: str, label_type: str = "act", 
                  max_length: Optional[int] = None, min_length: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load data as variable-length sequences (episodes) for temporal modeling.
    
    Args:
        files: List of NPZ file paths
        section: Control section ("hip", "knees", "both", "both-adv")
        label_type: Label type ("act" for actions, "qdes" for desired joint positions)
        max_length: Maximum sequence length (None for full episodes)
        min_length: Minimum sequence length to include
        
    Returns:
        List of (obs_seq, action_seq) tuples where each sequence is an episode
        
    Raises:
        ValueError: If section or label_type is invalid
    """
    if section not in SECTION_MAPPINGS:
        raise ValueError(f"Unknown section: {section}. Must be one of {list(SECTION_MAPPINGS.keys())}")
    
    if label_type not in ["act", "qdes"]:
        raise ValueError(f"Unknown label_type: {label_type}. Must be 'act' or 'qdes'")
    
    sequences = []
    
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
                    # Normalize qdes to [-1, 1] range for each joint
                    for i in range(labels.shape[1]):
                        lo, hi = JOINT_MIN[i], JOINT_MAX[i]
                        # Normalize: (value - lo) / (hi - lo) * 2 - 1 --> [-1, 1]
                        labels[:, i] = 2.0 * (labels[:, i] - lo) / (hi - lo) - 1.0
                else:
                    raise ValueError(f"No 'info_qdes' data in {file_path}")
            
            # Extract relevant joint indices
            joint_indices = SECTION_MAPPINGS[section]
            labels = labels[:, joint_indices]  # (T, n_joints)
            
            # Get episode length
            T = len(labels)
            
            # Filter by length
            if T < min_length:
                continue
            
            # Truncate if max_length specified
            if max_length is not None and T > max_length:
                T = max_length
            
            # Extract sequence
            obs_seq = obs[:T]  # (T, 11) - use first T observations
            action_seq = labels[:T]  # (T, n_joints)
            
            sequences.append((obs_seq, action_seq))
    
    if not sequences:
        raise ValueError("No valid sequences found")
    
    return sequences


def load_sequences_with_windows(files: List[str], section: str, label_type: str = "act",
                               window_size: int = 50, stride: int = 25, 
                               min_length: int = 10) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Load data as overlapping windows for temporal modeling.
    
    Args:
        files: List of NPZ file paths
        section: Control section
        label_type: Label type
        window_size: Size of temporal windows
        stride: Stride between windows
        min_length: Minimum episode length to include
        
    Returns:
        List of (obs_window, action_window) tuples
    """
    sequences = []
    
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
                    continue
            else:  # qdes
                if "info_qdes" in data:
                    labels = data["info_qdes"].astype(np.float32)  # (T, 3)
                    # Normalize qdes to [-1, 1] range for each joint
                    for i in range(labels.shape[1]):
                        lo, hi = JOINT_MIN[i], JOINT_MAX[i]
                        labels[:, i] = 2.0 * (labels[:, i] - lo) / (hi - lo) - 1.0
                else:
                    continue
            
            # Extract relevant joint indices
            joint_indices = SECTION_MAPPINGS[section]
            labels = labels[:, joint_indices]  # (T, n_joints)
            
            T = len(labels)
            
            # Filter by length
            if T < min_length:
                continue
            
            # Extract overlapping windows
            for start_idx in range(0, T - window_size + 1, stride):
                end_idx = start_idx + window_size
                
                obs_window = obs[start_idx:end_idx]  # (window_size, 11)
                action_window = labels[start_idx:end_idx]  # (window_size, n_joints)
                
                sequences.append((obs_window, action_window))
    
    return sequences


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for variable-length sequences.
    
    Handles padding and masking for temporal models.
    """
    
    def __init__(self, sequences: List[Tuple[np.ndarray, np.ndarray]], 
                 padding_strategy: str = "zero", max_length: Optional[int] = None):
        """
        Initialize sequence dataset.
        
        Args:
            sequences: List of (obs_seq, action_seq) tuples
            padding_strategy: "zero" or "last" (repeat last observation)
            max_length: Maximum sequence length (None for auto-detect)
        """
        self.sequences = sequences
        self.padding_strategy = padding_strategy
        
        # Compute max length if not provided
        if max_length is None:
            self.max_length = max(len(seq[0]) for seq in sequences)
        else:
            self.max_length = max_length
        
        # Compute sequence lengths
        self.sequence_lengths = [len(seq[0]) for seq in sequences]
        
        # Pad sequences
        self.padded_sequences = []
        for obs_seq, action_seq in sequences:
            padded_obs, padded_action = self._pad_sequence(obs_seq, action_seq)
            self.padded_sequences.append((padded_obs, padded_action))
    
    def _pad_sequence(self, obs_seq: np.ndarray, action_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad sequence to max_length."""
        seq_len = len(obs_seq)
        
        if seq_len >= self.max_length:
            # Truncate if too long
            padded_obs = obs_seq[:self.max_length]
            padded_action = action_seq[:self.max_length]
        else:
            # Pad if too short
            if self.padding_strategy == "zero":
                padded_obs = np.zeros((self.max_length, obs_seq.shape[1]), dtype=np.float32)
                padded_action = np.zeros((self.max_length, action_seq.shape[1]), dtype=np.float32)
                padded_obs[:seq_len] = obs_seq
                padded_action[:seq_len] = action_seq
            elif self.padding_strategy == "last":
                padded_obs = np.zeros((self.max_length, obs_seq.shape[1]), dtype=np.float32)
                padded_action = np.zeros((self.max_length, action_seq.shape[1]), dtype=np.float32)
                padded_obs[:seq_len] = obs_seq
                padded_action[:seq_len] = action_seq
                # Repeat last observation/action
                if seq_len > 0:
                    padded_obs[seq_len:] = obs_seq[-1]
                    padded_action[seq_len:] = action_seq[-1]
            else:
                raise ValueError(f"Unknown padding strategy: {self.padding_strategy}")
        
        return padded_obs, padded_action
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get item at index.
        
        Returns:
            Tuple of (padded_obs, padded_action, mask)
            mask: (max_length,) boolean tensor where True indicates valid timesteps
        """
        padded_obs, padded_action = self.padded_sequences[idx]
        seq_len = self.sequence_lengths[idx]
        
        # Create mask
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:seq_len] = True
        
        return (
            torch.tensor(padded_obs, dtype=torch.float32),
            torch.tensor(padded_action, dtype=torch.float32),
            mask
        )
    
    def get_sequence_lengths(self) -> List[int]:
        """Get list of sequence lengths."""
        return self.sequence_lengths.copy()


def create_sequence_loader(sequences: List[Tuple[np.ndarray, np.ndarray]], 
                          batch_size: int, shuffle: bool = True,
                          padding_strategy: str = "zero", 
                          max_length: Optional[int] = None,
                          num_workers: int = 0) -> DataLoader:
    """
    Create PyTorch DataLoader for sequences.
    
    Args:
        sequences: List of (obs_seq, action_seq) tuples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        padding_strategy: "zero" or "last"
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for sequences
    """
    dataset = SequenceDataset(sequences, padding_strategy, max_length)
    
    def collate_fn(batch):
        """Custom collate function for sequences."""
        obs_batch, action_batch, mask_batch = zip(*batch)
        
        # Stack tensors
        obs_tensor = torch.stack(obs_batch, dim=0)  # (batch, max_len, obs_dim)
        action_tensor = torch.stack(action_batch, dim=0)  # (batch, max_len, action_dim)
        mask_tensor = torch.stack(mask_batch, dim=0)  # (batch, max_len)
        
        return obs_tensor, action_tensor, mask_tensor
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )


def create_sequence_loader_from_files(files: List[str], section: str, batch_size: int,
                                     label_type: str = "act", shuffle: bool = True,
                                     max_length: Optional[int] = None,
                                     window_size: Optional[int] = None,
                                     stride: Optional[int] = None,
                                     padding_strategy: str = "zero",
                                     num_workers: int = 0) -> DataLoader:
    """
    Create sequence DataLoader directly from files.
    
    Args:
        files: List of NPZ file paths
        section: Control section
        label_type: Label type
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_length: Maximum sequence length
        window_size: If provided, use overlapping windows instead of full episodes
        stride: Stride for overlapping windows
        padding_strategy: "zero" or "last"
        num_workers: Number of worker processes
        
    Returns:
        DataLoader for sequences
    """
    if window_size is not None:
        # Use overlapping windows
        sequences = load_sequences_with_windows(
            files, section, label_type, window_size, stride or window_size//2
        )
    else:
        # Use full episodes
        sequences = load_sequences(files, section, label_type, max_length)
    
    return create_sequence_loader(
        sequences, batch_size, shuffle, padding_strategy, max_length, num_workers
    )


def validate_sequence_dataset(files: List[str], section: str, label_type: str = "act") -> Dict[str, any]:
    """
    Validate sequence dataset and return statistics.
    
    Args:
        files: List of NPZ file paths
        section: Control section
        label_type: Label type
        
    Returns:
        Dictionary with sequence dataset statistics
    """
    try:
        sequences = load_sequences(files, section, label_type)
        
        sequence_lengths = [len(seq[0]) for seq in sequences]
        obs_dims = [seq[0].shape[1] for seq in sequences]
        action_dims = [seq[1].shape[1] for seq in sequences]
        
        stats = {
            "n_sequences": len(sequences),
            "total_timesteps": sum(sequence_lengths),
            "avg_sequence_length": np.mean(sequence_lengths),
            "min_sequence_length": min(sequence_lengths),
            "max_sequence_length": max(sequence_lengths),
            "obs_dim": obs_dims[0] if obs_dims else 0,
            "action_dim": action_dims[0] if action_dims else 0,
            "section": section,
            "label_type": label_type
        }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}