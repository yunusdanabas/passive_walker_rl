"""
hip_knee_mse - Behaviour cloning for hip+knee controller with MSE loss.

This module implements a complete pipeline for training a neural network
to mimic a Finite State Machine (FSM) controller for the passive walker's hip and knee joints.
The pipeline consists of three main components:
1. Data collection from FSM demonstrations
2. Neural network training using MSE loss
3. Evaluation of the trained controller

Usage:
    python -m passive_walker.bc.hip_knee_mse.collect      --gpu  # Collect demonstration data
    python -m passive_walker.bc.hip_knee_mse.train        --gpu  # Train neural network
    python -m passive_walker.bc.hip_knee_mse.run_pipeline --gpu  # Run complete pipeline
"""

from pathlib import Path
import jax
import equinox as eqx
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController

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

DATA_BC_HIP_KNEE_MSE = DATA_BC / "hip_knee_mse"
DATA_BC_HIP_KNEE_MSE.mkdir(parents=True, exist_ok=True)

def save_model(model, model_file: Path):
    """
    Save a trained model to a file.

    Args:
        model: Trained model to save
        model_file: Path where to save the model
    """
    eqx.tree_serialise_leaves(model_file, model)


def load_model(path: Path,hidden_size=128,input_size=11):
    """
    Load model parameters from file.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model = HipKneeController(
        input_size=input_size,
        hidden_size=hidden_size,
        key=jax.random.PRNGKey(42)
    )
    return eqx.tree_deserialise_leaves(path, model)