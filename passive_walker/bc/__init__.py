# passive_walker/bc/__init__.py
"""
Behavior Cloning (BC) module for passive walker reinforcement learning.

This module provides configuration and utilities for Behavior Cloning (BC) seeded PPO algorithms,
including bc_init, scratch, and Brax implementations. It handles data storage paths,
model file management, and device selection functionality.

The module supports both standard BC and PPO-BC hybrid approaches for training
passive walker models.
"""

__version__ = "0.1.0"

# Standard library imports
from importlib import import_module
from typing import Optional

# Local imports
from passive_walker.constants import (
    ROOT,
    XML_PATH,
    DATA_DIR,
    DATA_BC,
    DATA_PPO_BC,
    RESULTS_BC,
    RESULTS_PPO_BC,
    RESULTS_PPO_SCRATCH,
    set_device
)

# Auto-import hip_mse module for direct execution
# This enables running `python -m passive_walker.bc.hip_mse.full_pipeline`
import_module("passive_walker.bc.hip_mse")

# Re-export commonly used constants and functions
__all__ = [
    'ROOT',
    'XML_PATH',
    'DATA_DIR',
    'DATA_BC',
    'DATA_PPO_BC',
    'RESULTS_BC',
    'RESULTS_PPO_BC',
    'RESULTS_PPO_SCRATCH',
    'set_device',
]

