"""
Multi-Task Learning for Behavior Cloning

Shared encoder with auxiliary tasks for improved representation learning.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from passive_walker.bc.models.temporal_torch import TorchLSTM, TorchGRU


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared encoder and task-specific heads.
    
    Main task: Action prediction
    Auxiliary tasks: Contact prediction, gait phase classification
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 use_temporal: bool = True,
                 temporal_type: str = "lstm"):
        """
        Initialize multi-task model.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_size: Hidden size for encoder
            num_layers: Number of layers
            dropout: Dropout probability
            use_temporal: Whether to use temporal encoder
            temporal_type: Type of temporal model ("lstm" or "gru")
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.use_temporal = use_temporal
        self.temporal_type = temporal_type
        
        # Shared encoder
        if use_temporal:
            if temporal_type == "lstm":
                self.encoder = TorchLSTM(
                    in_dim=obs_dim,
                    out_dim=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
            elif temporal_type == "gru":
                self.encoder = TorchGRU(
                    in_dim=obs_dim,
                    out_dim=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown temporal type: {temporal_type}")
        else:
            # MLP encoder
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Task-specific heads
        self.action_head = nn.Linear(hidden_size, action_dim)
        
        # Contact prediction head (binary classification for each foot)
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),  # left_contact, right_contact
            nn.Sigmoid()
        )
        
        # Gait phase classification head (4 phases: stance_left, swing_left, stance_right, swing_right)
        self.gait_phase_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 4),
            nn.Softmax(dim=-1)
        )
        
        # Velocity prediction head (regression)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)  # forward velocity
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Args:
            x: Input tensor (batch, features) or (batch, seq_len, features)
            hidden: Optional hidden state for temporal models
            
        Returns:
            Dictionary of task predictions
        """
        if self.use_temporal:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            # Temporal encoder
            encoded, new_hidden = self.encoder(x, hidden)
            
            # Use last timestep for predictions
            if encoded.dim() == 3:
                encoded = encoded[:, -1, :]  # (batch, hidden_size)
        else:
            # MLP encoder
            encoded = self.encoder(x)
            new_hidden = None
        
        # Task-specific predictions
        predictions = {
            "action": torch.tanh(self.action_head(encoded)),
            "contact": self.contact_head(encoded),
            "gait_phase": self.gait_phase_head(encoded),
            "velocity": self.velocity_head(encoded)
        }
        
        if new_hidden is not None:
            predictions["hidden"] = new_hidden
        
        return predictions
    
    def get_initial_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """Get initial hidden state for temporal models."""
        if self.use_temporal and hasattr(self.encoder, 'get_initial_hidden'):
            return self.encoder.get_initial_hidden(batch_size, device)
        return None


class MultiTaskLoss(nn.Module):
    """Multi-task loss function with learnable task weights."""
    
    def __init__(self, 
                 action_weight: float = 1.0,
                 contact_weight: float = 0.5,
                 gait_phase_weight: float = 0.3,
                 velocity_weight: float = 0.2,
                 learnable_weights: bool = True):
        """
        Initialize multi-task loss.
        
        Args:
            action_weight: Weight for action prediction loss
            contact_weight: Weight for contact prediction loss
            gait_phase_weight: Weight for gait phase classification loss
            velocity_weight: Weight for velocity prediction loss
            learnable_weights: Whether to use learnable task weights
        """
        super().__init__()
        
        self.learnable_weights = learnable_weights
        
        if learnable_weights:
            # Learnable log variances (inverse of weights)
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            # Fixed weights
            self.weights = {
                "action": action_weight,
                "contact": contact_weight,
                "gait_phase": gait_phase_weight,
                "velocity": velocity_weight
            }
        
        # Loss functions
        self.action_loss = nn.L1Loss()
        self.contact_loss = nn.BCELoss()
        self.gait_phase_loss = nn.CrossEntropyLoss()
        self.velocity_loss = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Action prediction loss
        if "action" in predictions and "action" in targets:
            action_loss = self.action_loss(predictions["action"], targets["action"])
            losses["action"] = action_loss
        
        # Contact prediction loss
        if "contact" in predictions and "contact" in targets:
            contact_loss = self.contact_loss(predictions["contact"], targets["contact"])
            losses["contact"] = contact_loss
        
        # Gait phase classification loss
        if "gait_phase" in predictions and "gait_phase" in targets:
            gait_phase_loss = self.gait_phase_loss(predictions["gait_phase"], targets["gait_phase"])
            losses["gait_phase"] = gait_phase_loss
        
        # Velocity prediction loss
        if "velocity" in predictions and "velocity" in targets:
            velocity_loss = self.velocity_loss(predictions["velocity"], targets["velocity"])
            losses["velocity"] = velocity_loss
        
        # Compute weighted total loss
        if self.learnable_weights:
            total_loss = 0.0
            loss_components = []
            
            for i, (task_name, loss_value) in enumerate(losses.items()):
                if i < len(self.log_vars):
                    # Use learnable weight: 1 / (2 * exp(log_var))
                    weight = 1.0 / (2.0 * torch.exp(self.log_vars[i]))
                    weighted_loss = weight * loss_value + 0.5 * self.log_vars[i]
                    total_loss += weighted_loss
                    loss_components.append(weighted_loss)
            
            losses["total"] = total_loss
            losses["loss_components"] = loss_components
        else:
            # Use fixed weights
            total_loss = 0.0
            for task_name, loss_value in losses.items():
                weight = self.weights.get(task_name, 1.0)
                total_loss += weight * loss_value
            
            losses["total"] = total_loss
        
        return losses


def extract_auxiliary_targets(obs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract auxiliary task targets from observations.
    
    Args:
        obs: Observation array (batch, obs_dim)
        
    Returns:
        Dictionary of auxiliary targets
    """
    targets = {}
    
    if obs.shape[1] >= 17:  # Enhanced observation space
        # Contact targets (binary)
        targets["contact"] = obs[:, 11:13]  # left_contact, right_contact
        
        # Gait phase targets (derived from contact pattern)
        left_contact = obs[:, 11]
        right_contact = obs[:, 12]
        
        # Simple gait phase classification
        gait_phase = np.zeros((obs.shape[0], 4))
        for i in range(obs.shape[0]):
            if left_contact[i] > 0.5 and right_contact[i] > 0.5:
                gait_phase[i, 0] = 1  # double support
            elif left_contact[i] > 0.5:
                gait_phase[i, 1] = 1  # left stance
            elif right_contact[i] > 0.5:
                gait_phase[i, 2] = 1  # right stance
            else:
                gait_phase[i, 3] = 1  # flight phase
        
        targets["gait_phase"] = gait_phase
        
        # Velocity target (forward velocity)
        targets["velocity"] = obs[:, 3:4]  # x velocity
    
    return targets


def create_multitask_dataset(obs_data: np.ndarray, action_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create multi-task dataset with auxiliary targets.
    
    Args:
        obs_data: Observation data
        action_data: Action data
        
    Returns:
        Dictionary of datasets
    """
    dataset = {
        "obs": obs_data,
        "action": action_data
    }
    
    # Extract auxiliary targets
    auxiliary_targets = extract_auxiliary_targets(obs_data)
    dataset.update(auxiliary_targets)
    
    return dataset


class MultiTaskTrainer:
    """Trainer for multi-task learning."""
    
    def __init__(self,
                 model: MultiTaskModel,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: MultiTaskLoss,
                 device: str = "cpu"):
        """
        Initialize multi-task trainer.
        
        Args:
            model: Multi-task model
            optimizer: Optimizer
            loss_fn: Multi-task loss function
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.model.to(device)
        self.loss_fn.to(device)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(batch["obs"])
        
        # Compute loss
        losses = self.loss_fn(predictions, batch)
        
        # Backward pass
        losses["total"].backward()
        self.optimizer.step()
        
        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        return loss_dict
    
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of losses
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(batch["obs"])
            losses = self.loss_fn(predictions, batch)
        
        # Convert to float for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        return loss_dict


def create_multitask_model(obs_dim: int, action_dim: int, **kwargs) -> MultiTaskModel:
    """Create multi-task model with default parameters."""
    return MultiTaskModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_size=kwargs.get("hidden_size", 128),
        num_layers=kwargs.get("num_layers", 1),
        dropout=kwargs.get("dropout", 0.1),
        use_temporal=kwargs.get("use_temporal", True),
        temporal_type=kwargs.get("temporal_type", "lstm")
    )


def create_multitask_loss(**kwargs) -> MultiTaskLoss:
    """Create multi-task loss with default parameters."""
    return MultiTaskLoss(
        action_weight=kwargs.get("action_weight", 1.0),
        contact_weight=kwargs.get("contact_weight", 0.5),
        gait_phase_weight=kwargs.get("gait_phase_weight", 0.3),
        velocity_weight=kwargs.get("velocity_weight", 0.2),
        learnable_weights=kwargs.get("learnable_weights", True)
    )
