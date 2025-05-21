# passive_walker/bc/hip_mse/__init__.py

"""
hip_mse - Behaviour cloning for hip-only controller with MSE loss.

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

DATA_BC_HIP_MSE = DATA_BC / "hip_mse"
DATA_BC_HIP_MSE.mkdir(parents=True, exist_ok=True)