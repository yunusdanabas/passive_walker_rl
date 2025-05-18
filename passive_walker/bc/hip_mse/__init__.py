# passive_walker/bc/hip_mse/__init__.py

"""
hip_mse – Behaviour cloning for hip-only controller with MSE loss.

This module implements a complete pipeline for training a neural network
to mimic a Finite State Machine (FSM) controller for the passive walker's hip joint.
The pipeline consists of three main components:
1. Data collection from FSM demonstrations
2. Neural network training using MSE loss
3. Evaluation of the trained controller

Usage:
    python -m passive_walker.bc.hip_mse.collect      --gpu  # Collect demonstration data
    python -m passive_walker.bc.hip_mse.train        --gpu  # Train neural network
    python -m passive_walker.bc.hip_mse.run_pipeline --gpu  # Run complete pipeline
"""

from pathlib import Path

# Directory setup
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "bc"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
XML_PATH = str(Path(__file__).resolve().parents[3] / "passiveWalker_model.xml")

def set_device(use_gpu: bool) -> None:
    """Configure JAX to use either GPU or CPU.
    
    Args:
        use_gpu: If True, use GPU; otherwise use CPU.
    """
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu" if use_gpu else "cpu")
