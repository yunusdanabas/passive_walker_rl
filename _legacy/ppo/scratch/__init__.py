"""
This module handles configuration for PPO algorithms from scratch.
It defines paths for data storage and model files, and provides device selection functionality.
"""

import os
import jax
import equinox as eqx
from pathlib import Path
from passive_walker.bc.hip_knee_mse import HipKneeController

# Root directory of the project
ROOT = Path(__file__).resolve().parents[3]

# Path to your MuJoCo XML
XML_PATH = ROOT / "models" / "passiveWalker_model.xml"

# Data directories
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# PPO-specific directories
PPO_DATA = DATA_DIR / "ppo"
PPO_DATA.mkdir(parents=True, exist_ok=True)

DATA_PPO_SCRATCH = PPO_DATA / "scratch"
DATA_PPO_SCRATCH.mkdir(parents=True, exist_ok=True)

# Results directories
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PPO_RESULTS = RESULTS_DIR / "ppo"
PPO_RESULTS.mkdir(parents=True, exist_ok=True)

RESULTS_PPO_SCRATCH = PPO_RESULTS / "scratch"
RESULTS_PPO_SCRATCH.mkdir(parents=True, exist_ok=True)

# Default hyperparameters
DEFAULT_STEPS = 4096
DEFAULT_SIGMA = 0.1
DEFAULT_HZ = 200
DEFAULT_ITERATIONS = 500
DEFAULT_ROLLOUT_STEPS = 8192
DEFAULT_PPO_EPOCHS = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_GAMMA = 0.99
DEFAULT_LAMBDA = 0.95
DEFAULT_CLIP_EPS = 0.2
DEFAULT_POLICY_LR = 1e-4
DEFAULT_CRITIC_LR = 1e-3

def set_device(use_gpu: bool):
    """
    Select JAX CPU vs GPU backend and configure device settings.
    
    Args:
        use_gpu: True to use GPU, False to use CPU
        
    Note:
        - Sets JAX platform name
        - Configures memory growth for GPU
        - Prints current device info
    """
    # Set JAX platform
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if use_gpu else "cpu"
    
    # Configure GPU memory growth if using GPU
    if use_gpu:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    # Print device info
    print(f"Using device: {jax.default_backend()}")
    if use_gpu:
        print(f"GPU available: {jax.devices('gpu')}")

def save_critic(critic, critic_file: Path):
    """
    Save a trained critic model to a file using tree_serialise_leaves.

    Args:
        critic: Trained critic model to save
        critic_file: Path where to save the critic model
    """
    eqx.tree_serialise_leaves(critic_file, critic)

def save_policy_and_critic(policy, critic, base_path: Path, hz: int = 0):
    """
    Save both policy and critic models to separate .eqx files.

    Args:
        policy: Trained policy model
        critic: Trained critic model
        base_path: Base directory to save the models
        hz: Control frequency used for training
    """
    # Create results directory if it doesn't exist
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save policy
    policy_path = base_path / f"policy_{hz}hz.eqx"
    eqx.tree_serialise_leaves(policy_path, policy)
    print(f"Saved policy → {policy_path}")
    
    # Save critic
    critic_path = base_path / f"critic_{hz}hz.eqx"
    eqx.tree_serialise_leaves(critic_path, critic)
    print(f"Saved critic → {critic_path}")

def save_model(model, model_file: Path):
    """
    Save a trained model to a file.

    Args:
        model: Trained model to save
        model_file: Path where to save the model
    """
    eqx.tree_serialise_leaves(model_file, model)

def load_model(path: Path, hidden_size=256, input_size=11):
    """
    Load model parameters from file.
    
    Args:
        path: Path to the saved model
        hidden_size: Size of hidden layers
        input_size: Size of input layer
        
    Returns:
        Loaded model
    """
    model = HipKneeController(
        input_size=input_size,
        hidden_size=hidden_size,
        key=jax.random.PRNGKey(42)
    )
    return eqx.tree_deserialise_leaves(path, model)
