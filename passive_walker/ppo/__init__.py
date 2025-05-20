"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms.
It defines paths for data storage and model files, and provides device selection functionality.
"""

import os
from pathlib import Path


ROOT       = Path(__file__).resolve().parents[2]
BC_DATA    = ROOT / "data" / "bc"

PPO_BC_DATA = ROOT / "data" / "ppo" / "bc_init"
PPO_BC_DATA.mkdir(parents=True, exist_ok=True)

# Where to read/write BC‐seeded PPO data
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ppo"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path to your MuJoCo XML
XML_PATH = Path(__file__).resolve().parents[2] / "models" / "passiveWalker_model.xml"

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"
