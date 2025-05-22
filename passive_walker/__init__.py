__version__ = "0.1.0"

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
import os
PPO_DIR = os.path.join(ROOT, "ppo")



def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"

