"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms.
It defines paths for data storage and model files, and provides device selection functionality.
"""

import os
from pathlib import Path

# Where to read/write scratch PPO data
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "ppo" / "scratch"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path to your MuJoCo XML
XML_PATH = Path(__file__).resolve().parents[3] / "passiveWalker_model.xml"

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"
