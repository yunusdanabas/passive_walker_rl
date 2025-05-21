"""
This module handles configuration for PPO algorithms from scratch.
It defines paths for data storage and model files, and provides device selection functionality.
"""

import os
from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[3]

# Path to your MuJoCo XML
XML_PATH = ROOT / "passiveWalker_model.xml"

# Data directories
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_PPO_SCRATCH = ROOT / "data" / "ppo" / "scratch"
DATA_PPO_SCRATCH.mkdir(parents=True, exist_ok=True)

RESULTS_PPO_SCRATCH = ROOT / "results" / "ppo" / "scratch"
RESULTS_PPO_SCRATCH.mkdir(parents=True, exist_ok=True)

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"
