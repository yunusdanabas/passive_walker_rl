"""
This module contains all constant definitions for the passive walker project,
including paths for data storage, model files, and other configuration settings.
"""

import os
from pathlib import Path

# Root directory of the project
ROOT = Path(__file__).resolve().parents[1]

# Path to MuJoCo XML
XML_PATH = ROOT / "passiveWalker_model.xml"

# Data directories
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DATA_BC = ROOT / "data" / "bc"
DATA_BC.mkdir(parents=True, exist_ok=True)

DATA_PPO_BC = ROOT / "data" / "ppo" / "bc_init"
DATA_PPO_BC.mkdir(parents=True, exist_ok=True)

# Results directories
RESULTS_BC = ROOT / "results" / "bc"
RESULTS_BC.mkdir(parents=True, exist_ok=True)

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
    device = "gpu" if use_gpu else "cpu"
    os.environ["JAX_PLATFORM_NAME"] = device
    print(f"Device set to: {device}")
