"""
Behavior Cloning (BC) module for passive walker.

This module provides both Torch and JAX backends for training neural networks
to mimic the FSM controller behavior.

Layout:
- config.py, utils.py: Core configuration and utilities
- training/: Training implementations
- data/: Data loading and augmentation
- evaluation/: Playback and evaluation tools
- models/: Model definitions
"""

# Core imports that are safe and commonly used
from passive_walker.bc.config import (
    TrainingConfig,
    TemporalTrainingConfig,
    EvaluationConfig,
    create_default_config,
    create_research_config
)

from passive_walker.bc.utils import (
    set_seed,
    Normalizer,
    save_checkpoint,
    load_checkpoint
)

# For other imports, users should import from submodules directly:
# from passive_walker.bc.training import train_torch, train_jax
# from passive_walker.bc.data import SequenceDataset, discover_npzs
# from passive_walker.bc.evaluation import ComprehensiveEvaluator
# etc.

__all__ = [
    # Config
    "TrainingConfig",
    "TemporalTrainingConfig",
    "EvaluationConfig",
    "create_default_config",
    "create_research_config",
    # Utils
    "set_seed",
    "Normalizer",
    "save_checkpoint",
    "load_checkpoint",
]
