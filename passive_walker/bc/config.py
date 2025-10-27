"""
Structured Configuration Validation for BC Training

Provides dataclass-based configuration with validation for robust training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
import json

from passive_walker.config.paths import BC_MODELS_DIR, BC_RUNS_DIR, METRICS_DIR
from passive_walker.config.paths_redirect import redirect_legacy_dir


@dataclass
class TrainingConfig:
    """Configuration for BC training with validation."""
    
    # Core training parameters
    backend: str = "torch"  # "torch" or "jax"
    section: str = "both"   # "hip", "knees", or "both"
    data_dir: str = "experiments/data/fsm_demos"
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    
    # Environment parameters
    ctrl_hz: int = 100
    use_enhanced_rewards: bool = False
    randomization_profile: Optional[str] = None
    
    # Training infrastructure
    scheduler: str = "none"  # "none", "plateau", "cosine"
    augment: bool = False
    validation_split: float = 0.2
    
    # Model parameters
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    dropout: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: str = str(BC_MODELS_DIR)
    log_dir: str = str(BC_RUNS_DIR)
    
    # Validation parameters
    validate_every: int = 5
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_core_params()
        self._validate_paths()
        self._validate_model_params()
        self._validate_training_params()
    
    def _validate_core_params(self):
        """Validate core training parameters."""
        if self.backend not in ["torch", "jax"]:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'torch' or 'jax'")
        
        if self.section not in ["hip", "knees", "both"]:
            raise ValueError(f"Invalid section: {self.section}. Must be 'hip', 'knees', or 'both'")
        
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}. Must be positive")
        
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}. Must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}. Must be positive")
        
        if self.ctrl_hz not in [100, 150, 200]:
            raise ValueError(f"Invalid ctrl_hz: {self.ctrl_hz}. Must be 100, 150, or 200")
    
    def _validate_paths(self):
        """Validate file paths."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _validate_model_params(self):
        """Validate model parameters."""
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        
        for size in self.hidden_sizes:
            if size <= 0:
                raise ValueError(f"Invalid hidden size: {size}. Must be positive")
        
        if self.activation not in ["relu", "tanh", "sigmoid", "gelu"]:
            raise ValueError(f"Invalid activation: {self.activation}")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Invalid dropout: {self.dropout}. Must be between 0 and 1")
    
    def _validate_training_params(self):
        """Validate training parameters."""
        if self.scheduler not in ["none", "plateau", "cosine"]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Invalid validation_split: {self.validation_split}. Must be between 0 and 1")
        
        if self.log_interval <= 0:
            raise ValueError(f"Invalid log_interval: {self.log_interval}. Must be positive")
        
        if self.save_interval <= 0:
            raise ValueError(f"Invalid save_interval: {self.save_interval}. Must be positive")
        
        if self.validate_every <= 0:
            raise ValueError(f"Invalid validate_every: {self.validate_every}. Must be positive")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"Invalid early_stopping_patience: {self.early_stopping_patience}. Must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "backend": self.backend,
            "section": self.section,
            "data_dir": self.data_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "ctrl_hz": self.ctrl_hz,
            "use_enhanced_rewards": self.use_enhanced_rewards,
            "randomization_profile": self.randomization_profile,
            "scheduler": self.scheduler,
            "augment": self.augment,
            "validation_split": self.validation_split,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "dropout": self.dropout,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "validate_every": self.validate_every,
            "early_stopping_patience": self.early_stopping_patience,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> TrainingConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_args(cls, args) -> TrainingConfig:
        """Create configuration from command line arguments."""
        return cls(
            backend=getattr(args, 'backend', 'torch'),
            section=getattr(args, 'section', 'both'),
            data_dir=getattr(args, 'data_dir', 'experiments/data/fsm_demos'),
            epochs=getattr(args, 'epochs', 100),
            batch_size=getattr(args, 'batch_size', 64),
            learning_rate=getattr(args, 'learning_rate', 1e-3),
            seed=getattr(args, 'seed', 42),
            ctrl_hz=getattr(args, 'ctrl_hz', 100),
            use_enhanced_rewards=getattr(args, 'use_enhanced_rewards', False),
            randomization_profile=getattr(args, 'randomization_profile', None),
            scheduler=getattr(args, 'scheduler', 'none'),
            augment=getattr(args, 'augment', False),
            validation_split=getattr(args, 'validation_split', 0.2),
            hidden_sizes=getattr(args, 'hidden_sizes', [256, 256]),
            activation=getattr(args, 'activation', 'relu'),
            dropout=getattr(args, 'dropout', 0.0),
            log_interval=getattr(args, 'log_interval', 10),
            save_interval=getattr(args, 'save_interval', 50),
            checkpoint_dir=getattr(args, 'checkpoint_dir', str(BC_MODELS_DIR)),
            validate_every=getattr(args, 'validate_every', 5),
            early_stopping_patience=getattr(args, 'early_stopping_patience', 20),
        )


@dataclass
class TemporalTrainingConfig:
    """Configuration for temporal BC training with validation."""
    
    # Core training parameters
    backend: str = "torch"  # "torch" or "jax"
    section: str = "both"   # "hip", "knees", or "both"
    data_dir: str = "experiments/data/fsm_demos"
    epochs: int = 100
    batch_size: int = 32  # Smaller batch size for sequences
    learning_rate: float = 1e-3
    seed: int = 42
    
    # Environment parameters
    ctrl_hz: int = 100
    use_enhanced_rewards: bool = False
    randomization_profile: Optional[str] = None
    
    # Training infrastructure
    scheduler: str = "none"  # "none", "plateau", "cosine"
    augment: bool = False
    validation_split: float = 0.2
    
    # Temporal model parameters
    model_type: str = "lstm"  # "lstm", "gru", "bilstm"
    hidden_size: int = 128
    num_layers: int = 1
    bidirectional: bool = False
    dropout: float = 0.1
    
    # Sequence parameters
    sequence_length: Optional[int] = None  # None for full episodes
    window_size: Optional[int] = None  # None for full episodes, int for windows
    stride: Optional[int] = None  # Stride for windows
    padding_strategy: str = "zero"  # "zero" or "last"
    
    # Temporal augmentation
    temporal_augmentation: bool = False
    augmentation_type: str = "default"  # "light", "default", "heavy"
    
    # Loss parameters
    loss_type: str = "l1"  # "l1", "mse", "smooth_l1"
    temporal_smoothness_weight: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: str = str(BC_MODELS_DIR)
    
    # Validation parameters
    validate_every: int = 5
    early_stopping_patience: int = 20
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_core_params()
        self._validate_paths()
        self._validate_temporal_params()
        self._validate_training_params()
    
    def _validate_core_params(self):
        """Validate core training parameters."""
        if self.backend not in ["torch", "jax"]:
            raise ValueError(f"Invalid backend: {self.backend}. Must be 'torch' or 'jax'")
        
        if self.section not in ["hip", "knees", "both", "both-adv"]:
            raise ValueError(f"Invalid section: {self.section}. Must be 'hip', 'knees', 'both', or 'both-adv'")
        
        if self.epochs <= 0:
            raise ValueError(f"Invalid epochs: {self.epochs}. Must be positive")
        
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}. Must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning_rate: {self.learning_rate}. Must be positive")
        
        if self.ctrl_hz not in [100, 150, 200]:
            raise ValueError(f"Invalid ctrl_hz: {self.ctrl_hz}. Must be 100, 150, or 200")
    
    def _validate_paths(self):
        """Validate file paths."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _validate_temporal_params(self):
        """Validate temporal model parameters."""
        if self.model_type not in ["lstm", "gru", "bilstm"]:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'lstm', 'gru', or 'bilstm'")
        
        if self.hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {self.hidden_size}. Must be positive")
        
        if self.num_layers <= 0:
            raise ValueError(f"Invalid num_layers: {self.num_layers}. Must be positive")
        
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"Invalid dropout: {self.dropout}. Must be between 0 and 1")
        
        if self.padding_strategy not in ["zero", "last"]:
            raise ValueError(f"Invalid padding_strategy: {self.padding_strategy}. Must be 'zero' or 'last'")
        
        if self.augmentation_type not in ["light", "default", "heavy"]:
            raise ValueError(f"Invalid augmentation_type: {self.augmentation_type}. Must be 'light', 'default', or 'heavy'")
        
        if self.loss_type not in ["l1", "mse", "smooth_l1"]:
            raise ValueError(f"Invalid loss_type: {self.loss_type}. Must be 'l1', 'mse', or 'smooth_l1'")
        
        if self.temporal_smoothness_weight < 0:
            raise ValueError(f"Invalid temporal_smoothness_weight: {self.temporal_smoothness_weight}. Must be non-negative")
        
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"Invalid gradient_clip_norm: {self.gradient_clip_norm}. Must be positive")
    
    def _validate_training_params(self):
        """Validate training parameters."""
        if self.scheduler not in ["none", "plateau", "cosine"]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        
        if not 0 < self.validation_split < 1:
            raise ValueError(f"Invalid validation_split: {self.validation_split}. Must be between 0 and 1")
        
        if self.log_interval <= 0:
            raise ValueError(f"Invalid log_interval: {self.log_interval}. Must be positive")
        
        if self.save_interval <= 0:
            raise ValueError(f"Invalid save_interval: {self.save_interval}. Must be positive")
        
        if self.validate_every <= 0:
            raise ValueError(f"Invalid validate_every: {self.validate_every}. Must be positive")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"Invalid early_stopping_patience: {self.early_stopping_patience}. Must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "backend": self.backend,
            "section": self.section,
            "data_dir": self.data_dir,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "ctrl_hz": self.ctrl_hz,
            "use_enhanced_rewards": self.use_enhanced_rewards,
            "randomization_profile": self.randomization_profile,
            "scheduler": self.scheduler,
            "augment": self.augment,
            "validation_split": self.validation_split,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
            "sequence_length": self.sequence_length,
            "window_size": self.window_size,
            "stride": self.stride,
            "padding_strategy": self.padding_strategy,
            "temporal_augmentation": self.temporal_augmentation,
            "augmentation_type": self.augmentation_type,
            "loss_type": self.loss_type,
            "temporal_smoothness_weight": self.temporal_smoothness_weight,
            "gradient_clip_norm": self.gradient_clip_norm,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "validate_every": self.validate_every,
            "early_stopping_patience": self.early_stopping_patience,
        }
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> TemporalTrainingConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_args(cls, args) -> TemporalTrainingConfig:
        """Create configuration from command line arguments."""
        return cls(
            backend=getattr(args, 'backend', 'torch'),
            section=getattr(args, 'section', 'both'),
            data_dir=getattr(args, 'data_dir', 'experiments/data/fsm_demos'),
            epochs=getattr(args, 'epochs', 100),
            batch_size=getattr(args, 'batch_size', 32),
            learning_rate=getattr(args, 'learning_rate', 1e-3),
            seed=getattr(args, 'seed', 42),
            ctrl_hz=getattr(args, 'ctrl_hz', 100),
            use_enhanced_rewards=getattr(args, 'use_enhanced_rewards', False),
            randomization_profile=getattr(args, 'randomization_profile', None),
            scheduler=getattr(args, 'scheduler', 'none'),
            augment=getattr(args, 'augment', False),
            validation_split=getattr(args, 'validation_split', 0.2),
            model_type=getattr(args, 'model_type', 'lstm'),
            hidden_size=getattr(args, 'hidden_size', 128),
            num_layers=getattr(args, 'num_layers', 1),
            bidirectional=getattr(args, 'bidirectional', False),
            dropout=getattr(args, 'dropout', 0.1),
            sequence_length=getattr(args, 'sequence_length', None),
            window_size=getattr(args, 'window_size', None),
            stride=getattr(args, 'stride', None),
            padding_strategy=getattr(args, 'padding_strategy', 'zero'),
            temporal_augmentation=getattr(args, 'temporal_augmentation', False),
            augmentation_type=getattr(args, 'augmentation_type', 'default'),
            loss_type=getattr(args, 'loss_type', 'l1'),
            temporal_smoothness_weight=getattr(args, 'temporal_smoothness_weight', 0.1),
            gradient_clip_norm=getattr(args, 'gradient_clip_norm', 1.0),
            log_interval=getattr(args, 'log_interval', 10),
            save_interval=getattr(args, 'save_interval', 50),
            checkpoint_dir=getattr(args, 'checkpoint_dir', str(BC_MODELS_DIR)),
            validate_every=getattr(args, 'validate_every', 5),
            early_stopping_patience=getattr(args, 'early_stopping_patience', 20),
        )


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Model and data
    checkpoint_path: str
    data_dir: str = "experiments/data/fsm_demos"
    backend: str = "torch"
    
    # Evaluation parameters
    episodes: int = 10
    duration_sec: float = 25.0
    physics_conditions: List[str] = field(default_factory=lambda: ["nominal"])
    
    # Environment parameters
    ctrl_hz: int = 100
    use_enhanced_rewards: bool = False
    randomization_profile: Optional[str] = None
    
    # Output
    output_dir: str = str(METRICS_DIR / "bc")
    save_trajectories: bool = True
    generate_plots: bool = True
    
    def __post_init__(self):
        """Validate evaluation configuration."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if self.backend not in ["torch", "jax"]:
            raise ValueError(f"Invalid backend: {self.backend}")
        
        if self.episodes <= 0:
            raise ValueError(f"Invalid episodes: {self.episodes}")
        
        if self.duration_sec <= 0:
            raise ValueError(f"Invalid duration_sec: {self.duration_sec}")
        
        # Redirect legacy output paths and ensure directory exists
        self.output_dir = str(redirect_legacy_dir(self.output_dir))
        os.makedirs(self.output_dir, exist_ok=True)


def create_default_config() -> TrainingConfig:
    """Create a default training configuration."""
    return TrainingConfig()


def create_research_config() -> TrainingConfig:
    """Create a configuration optimized for research/RL training."""
    return TrainingConfig(
        use_enhanced_rewards=True,
        randomization_profile="moderate",
        scheduler="cosine",
        augment=True,
        hidden_sizes=[512, 512, 256],
        dropout=0.1,
        early_stopping_patience=30,
    )

