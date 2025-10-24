"""
PPO Actor-Critic Models

Simple actor-critic architectures for PPO training.
Supports MLP, LSTM, and GRU with proper action distributions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class ActorCriticMLP(nn.Module):
    """
    Simple MLP actor-critic for PPO.
    
    Shared encoder with separate actor and critic heads.
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 hidden_sizes: list[int] = [64, 64],
                 activation: str = "tanh",
                 use_orthogonal_init: bool = True):
        """
        Initialize MLP actor-critic.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function ("tanh", "relu")
            use_orthogonal_init: Use orthogonal initialization
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        
        # Activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared encoder
        self.encoder = self._build_encoder()
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self._orthogonal_init()
    
    def _build_encoder(self) -> nn.Module:
        """Build shared encoder network."""
        layers = []
        input_dim = self.obs_dim
        
        for hidden_size in self.hidden_sizes:
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.Tanh()
            ])
            input_dim = hidden_size
        
        return nn.Sequential(*layers)
    
    def _orthogonal_init(self):
        """Initialize weights orthogonally."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Special initialization for actor mean
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        # Special initialization for critic
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observations
            
        Returns:
            action_mean: Action means
            action_log_std: Action log standard deviations
            value: State values
        """
        # Shared encoding
        features = self.encoder(obs)
        
        # Actor head
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic head
        value = self.critic(features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            obs: Observations
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Sampled actions
            log_prob: Log probabilities
            value: State values
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            # Sample from Gaussian distribution
            action_std = torch.exp(action_log_std)
            normal = torch.distributions.Normal(action_mean, action_std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss computation.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            log_prob: Log probabilities
            entropy: Action entropy
            value: State values
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        # Compute log probability
        action_std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value


class ActorCriticLSTM(nn.Module):
    """
    LSTM-based actor-critic for PPO.
    
    Uses LSTM for temporal modeling with separate actor/critic heads.
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 use_orthogonal_init: bool = True):
        """
        Initialize LSTM actor-critic.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            use_orthogonal_init: Use orthogonal initialization
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(obs_dim, hidden_size, num_layers, batch_first=True)
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self._orthogonal_init()
    
    def _orthogonal_init(self):
        """Initialize weights orthogonally."""
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize actor/critic heads
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with LSTM.
        
        Args:
            obs: Observations (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: LSTM hidden state
            
        Returns:
            action_mean: Action means
            action_log_std: Action log standard deviations
            value: State values
            hidden: Updated LSTM hidden state
        """
        # Handle single timestep input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(obs, hidden)
        
        # Use last timestep output
        features = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Actor head
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic head
        value = self.critic(features)
        
        return action_mean, action_log_std, value, hidden
    
    def get_action(self, obs: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get action from LSTM policy.
        
        Args:
            obs: Observations
            hidden: LSTM hidden state
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Sampled actions
            log_prob: Log probabilities
            value: State values
            hidden: Updated LSTM hidden state
        """
        action_mean, action_log_std, value, hidden = self.forward(obs, hidden)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            # Sample from Gaussian distribution
            action_std = torch.exp(action_log_std)
            normal = torch.distributions.Normal(action_mean, action_std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evaluate actions for PPO loss computation.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            hidden: LSTM hidden state
            
        Returns:
            log_prob: Log probabilities
            entropy: Action entropy
            value: State values
            hidden: Updated LSTM hidden state
        """
        action_mean, action_log_std, value, hidden = self.forward(obs, hidden)
        
        # Compute log probability
        action_std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value, hidden


class ActorCriticGRU(nn.Module):
    """
    GRU-based actor-critic for PPO.
    
    Uses GRU for temporal modeling with separate actor/critic heads.
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 use_orthogonal_init: bool = True):
        """
        Initialize GRU actor-critic.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_size: GRU hidden size
            num_layers: Number of GRU layers
            use_orthogonal_init: Use orthogonal initialization
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU encoder
        self.gru = nn.GRU(obs_dim, hidden_size, num_layers, batch_first=True)
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self._orthogonal_init()
    
    def _orthogonal_init(self):
        """Initialize weights orthogonally."""
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Initialize actor/critic heads
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with GRU.
        
        Args:
            obs: Observations (batch_size, seq_len, obs_dim) or (batch_size, obs_dim)
            hidden: GRU hidden state
            
        Returns:
            action_mean: Action means
            action_log_std: Action log standard deviations
            value: State values
            hidden: Updated GRU hidden state
        """
        # Handle single timestep input
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
        
        # GRU forward pass
        gru_out, hidden = self.gru(obs, hidden)
        
        # Use last timestep output
        features = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Actor head
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic head
        value = self.critic(features)
        
        return action_mean, action_log_std, value, hidden
    
    def get_action(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from GRU policy.
        
        Args:
            obs: Observations
            hidden: GRU hidden state
            deterministic: Whether to use deterministic action
            
        Returns:
            action: Sampled actions
            log_prob: Log probabilities
            value: State values
            hidden: Updated GRU hidden state
        """
        action_mean, action_log_std, value, hidden = self.forward(obs, hidden)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean)
        else:
            # Sample from Gaussian distribution
            action_std = torch.exp(action_log_std)
            normal = torch.distributions.Normal(action_mean, action_std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value, hidden
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO loss computation.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            hidden: GRU hidden state
            
        Returns:
            log_prob: Log probabilities
            entropy: Action entropy
            value: State values
            hidden: Updated GRU hidden state
        """
        action_mean, action_log_std, value, hidden = self.forward(obs, hidden)
        
        # Compute log probability
        action_std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, action_std)
        log_prob = normal.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Compute entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value, hidden


def create_actor_critic(model_type: str, obs_dim: int, action_dim: int, **kwargs) -> nn.Module:
    """
    Create actor-critic model.
    
    Args:
        model_type: Model type ("mlp", "lstm", "gru")
        obs_dim: Observation dimension
        action_dim: Action dimension
        **kwargs: Additional model arguments
        
    Returns:
        Actor-critic model
    """
    if model_type == "mlp":
        return ActorCriticMLP(obs_dim, action_dim, **kwargs)
    elif model_type == "lstm":
        return ActorCriticLSTM(obs_dim, action_dim, **kwargs)
    elif model_type == "gru":
        return ActorCriticGRU(obs_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_bc_weights(actor_critic: nn.Module, bc_model_path: str, actor_only: bool = True):
    """
    Load BC model weights into actor-critic.
    
    Args:
        actor_critic: Actor-critic model
        bc_model_path: Path to BC model checkpoint
        actor_only: Whether to load only actor weights
    """
    bc_checkpoint = torch.load(bc_model_path, map_location="cpu")
    
    if isinstance(actor_critic, ActorCriticMLP):
        # Load BC weights into actor mean
        if "model_state_dict" in bc_checkpoint:
            bc_state = bc_checkpoint["model_state_dict"]
        else:
            bc_state = bc_checkpoint
        
        # Match BC model structure to actor head
        actor_critic.actor_mean.load_state_dict({
            "weight": bc_state["weight"],
            "bias": bc_state["bias"]
        })
        
        print(f"Loaded BC weights into MLP actor")
    
    elif isinstance(actor_critic, (ActorCriticLSTM, ActorCriticGRU)):
        # For temporal models, load BC weights into actor mean
        if "model_state_dict" in bc_checkpoint:
            bc_state = bc_checkpoint["model_state_dict"]
        else:
            bc_state = bc_checkpoint
        
        # Match BC model structure to actor head
        actor_critic.actor_mean.load_state_dict({
            "weight": bc_state["weight"],
            "bias": bc_state["bias"]
        })
        
        print(f"Loaded BC weights into {type(actor_critic).__name__} actor")
    
    else:
        raise ValueError(f"Unknown actor-critic type: {type(actor_critic)}")
