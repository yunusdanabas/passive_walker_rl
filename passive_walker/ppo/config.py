"""
PPO Configuration

Simple configuration system for PPO training hyperparameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
import os


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    
    # Experiment metadata
    experiment_name: str
    description: str = ""
    
    # Model configuration
    model_type: str = "mlp"  # "mlp", "lstm", "gru"
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])  # For MLP
    hidden_size: int = 64  # For LSTM/GRU
    num_layers: int = 1  # For LSTM/GRU
    
    # Training configuration
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_epochs: int = 10
    batch_size: int = 64
    n_steps: int = 2048  # Steps per rollout
    
    # PPO specific
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Environment configuration
    env_name: str = "PassiveWalkerEnv"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_envs: int = 8  # Number of parallel environments
    
    # Enhanced environment features
    use_curriculum: bool = False
    use_domain_randomization: bool = False
    randomization_profile: str = "moderate"  # "light", "moderate", "aggressive"
    
    # Evaluation configuration
    eval_freq: int = 25000  # Evaluate every 25k steps (more frequent for better tracking)
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Logging configuration
    log_freq: int = 1000
    save_freq: int = 50000
    log_dir: str = "experiments/runs/ppo"
    
    # BC initialization
    bc_model_path: Optional[str] = None
    bc_init_epochs: int = 0  # Epochs to train with BC loss only
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate model type
        if self.model_type not in ["mlp", "lstm", "gru"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        # Validate environment kwargs
        if not isinstance(self.env_kwargs, dict):
            raise ValueError("env_kwargs must be a dictionary")
        
        # Validate PPO parameters
        if not 0 < self.gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        
        if not 0 < self.gae_lambda <= 1:
            raise ValueError(f"gae_lambda must be in (0, 1], got {self.gae_lambda}")
        
        if not 0 < self.clip_range < 1:
            raise ValueError(f"clip_range must be in (0, 1), got {self.clip_range}")
        
        if self.value_loss_coef < 0:
            raise ValueError(f"value_loss_coef must be >= 0, got {self.value_loss_coef}")
        
        if self.entropy_coef < 0:
            raise ValueError(f"entropy_coef must be >= 0, got {self.entropy_coef}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "description": self.description,
            "model_type": self.model_type,
            "hidden_sizes": self.hidden_sizes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "env_name": self.env_name,
            "env_kwargs": self.env_kwargs,
            "n_envs": self.n_envs,
            "eval_freq": self.eval_freq,
            "n_eval_episodes": self.n_eval_episodes,
            "eval_deterministic": self.eval_deterministic,
            "log_freq": self.log_freq,
            "save_freq": self.save_freq,
            "log_dir": self.log_dir,
            "bc_model_path": self.bc_model_path,
            "bc_init_epochs": self.bc_init_epochs,
            "use_curriculum": self.use_curriculum,
            "use_domain_randomization": self.use_domain_randomization,
            "randomization_profile": self.randomization_profile
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> PPOConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


def create_default_configs() -> Dict[str, PPOConfig]:
    """Create default PPO configurations."""
    configs = {}
    
    # Basic MLP PPO
    configs["ppo_mlp_basic"] = PPOConfig(
        experiment_name="ppo_mlp_basic",
        description="Basic MLP PPO with default hyperparameters",
        model_type="mlp",
        hidden_sizes=[64, 64],
        total_timesteps=500_000,
        n_envs=4
    )
    
    # LSTM PPO
    configs["ppo_lstm_temporal"] = PPOConfig(
        experiment_name="ppo_lstm_temporal",
        description="LSTM PPO for temporal modeling",
        model_type="lstm",
        hidden_size=128,
        num_layers=2,
        total_timesteps=1_000_000,
        n_envs=8
    )
    
    # GRU PPO
    configs["ppo_gru_temporal"] = PPOConfig(
        experiment_name="ppo_gru_temporal",
        description="GRU PPO for temporal modeling",
        model_type="gru",
        hidden_size=128,
        num_layers=2,
        total_timesteps=1_000_000,
        n_envs=8
    )
    
    # BC-initialized PPO
    configs["ppo_bc_init"] = PPOConfig(
        experiment_name="ppo_bc_init",
        description="PPO initialized with BC model",
        model_type="mlp",
        hidden_sizes=[128, 128],
        bc_model_path="experiments/models/bc/best_model.pth",
        bc_init_epochs=5,
        total_timesteps=1_000_000,
        n_envs=8
    )
    
    # Advanced PPO with curriculum
    configs["ppo_advanced"] = PPOConfig(
        experiment_name="ppo_advanced",
        description="Advanced PPO with curriculum and domain randomization",
        model_type="lstm",
        hidden_size=128,
        num_layers=2,
        use_curriculum=True,
        use_domain_randomization=True,
        randomization_profile="aggressive",
        total_timesteps=2_000_000,
        n_envs=16
    )
    
    return configs


def create_config_from_args(args) -> PPOConfig:
    """Create PPO config from command line arguments."""
    return PPOConfig(
        experiment_name=args.experiment_name,
        description=args.description,
        model_type=args.model_type,
        hidden_sizes=args.hidden_sizes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        log_dir=args.log_dir,
        bc_model_path=args.bc_model_path,
        bc_init_epochs=args.bc_init_epochs,
        use_curriculum=args.use_curriculum,
        use_domain_randomization=args.use_domain_randomization,
        randomization_profile=args.randomization_profile
    )
