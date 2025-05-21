# passive_walker/bc/__init__.py
__version__ = "0.1.0"

from importlib import import_module

# auto-import hip_mse so `python -m passive_walker.bc.hip_mse.full_pipeline` works
import_module("passive_walker.bc.hip_mse")

"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms (bc_init and scratch and Brax).
It defines paths for data storage and model files, and provides device selection functionality.
"""

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

