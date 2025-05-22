"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms.
It defines paths for data storage, model files, and results, and provides device selection functionality.
"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]

# BC-related paths
BC_DATA = ROOT / "data" / "bc"
BC_RESULTS = ROOT / "results" / "bc"

# PPO-related paths
PPO_DATA = ROOT / "data" / "ppo"
PPO_RESULTS = ROOT / "results" / "ppo"

# BC-seeded PPO specific paths
PPO_BC_DATA = PPO_DATA / "bc_init"
PPO_BC_RESULTS = PPO_RESULTS / "bc_init"

# Create necessary directories
PPO_DATA.mkdir(parents=True, exist_ok=True)
PPO_RESULTS.mkdir(parents=True, exist_ok=True)
PPO_BC_DATA.mkdir(parents=True, exist_ok=True)
PPO_BC_RESULTS.mkdir(parents=True, exist_ok=True)

# Path to your MuJoCo XML
XML_PATH = ROOT / "models" / "passiveWalker_model.xml"

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
    """
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"
