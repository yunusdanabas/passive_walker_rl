"""
This module handles configuration for Behavior Cloning (BC) seeded PPO algorithms.
It defines paths for data storage, model files, and results, and provides device selection functionality.
"""

from pathlib import Path
import os
import jax
import equinox as eqx
from passive_walker.bc.hip_knee_mse import HipKneeController

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