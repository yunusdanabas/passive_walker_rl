__version__ = "0.1.0"

"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms (bc_init and scratch and Brax).
It defines paths for data storage and model files, and provides device selection functionality.
"""

import os
from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[2]

# Path to your MuJoCo XML
XML_PATH = ROOT / "passiveWalker_model.xml"

# Data directories
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BC_DATA = ROOT / "data" / "bc"
BC_DATA.mkdir(parents=True, exist_ok=True)

DATA_PPO_BC = ROOT / "data" / "ppo" / "bc_init"
DATA_PPO_BC.mkdir(parents=True, exist_ok=True)

RESULTS_PPO_BC = ROOT / "results" / "ppo" / "bc_init"
RESULTS_PPO_BC.mkdir(parents=True, exist_ok=True)

RESULTS_PPO_SCRATCH = ROOT / "results" / "ppo" / "scratch"
RESULTS_PPO_SCRATCH.mkdir(parents=True, exist_ok=True)

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"

