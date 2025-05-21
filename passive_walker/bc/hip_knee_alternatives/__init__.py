"""
hip_knee_alternatives - Behaviour cloning for hip+knee controller with various loss functions.

This module implements a complete pipeline for training neural networks
to mimic a Finite State Machine (FSM) controller for the passive walker's hip and knee joints
using different loss functions (MSE, Huber, L1, Combined).

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.collect              --gpu  # Collect demonstration data
    python -m passive_walker.bc.hip_knee_alternatives.train_mse           --gpu  # Train with MSE loss
    python -m passive_walker.bc.hip_knee_alternatives.train_huber         --gpu  # Train with Huber loss
    python -m passive_walker.bc.hip_knee_alternatives.train_l1            --gpu  # Train with L1 loss
    python -m passive_walker.bc.hip_knee_alternatives.train_combined      --gpu  # Train with combined loss
    python -m passive_walker.bc.hip_knee_alternatives.run_comparison_pipeline --gpu  # Run complete comparison
"""

import pickle
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from pathlib import Path
import numpy as np

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
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController

DATA_BC_HIP_KNEE_ALTERNATIVES = DATA_BC / "hip_knee_alternatives"
DATA_BC_HIP_KNEE_ALTERNATIVES.mkdir(parents=True, exist_ok=True)

MODEL_BC_HIP_KNEE_ALTERNATIVES = DATA_BC_HIP_KNEE_ALTERNATIVES / "models"
MODEL_BC_HIP_KNEE_ALTERNATIVES.mkdir(parents=True, exist_ok=True)


def set_device(use_gpu: bool):
    """Set JAX device based on availability and preference."""
    if use_gpu:
        try:
            jax.config.update("jax_platform_name", "gpu")
            print("[bc] Using GPU")
        except RuntimeError:
            print("[bc] GPU not available, using CPU")
    else:
        jax.config.update("jax_platform_name", "cpu")
        print("[bc] Using CPU")


def load_demo_data(demo_file: Path):
    """
    Load demonstration data from a pickle file.

    Args:
        demo_file: Path to the pickle file containing demonstration data

    Returns:
        obs: jnp.ndarray of shape (num_steps, obs_dim)
        labels: jnp.ndarray of shape (num_steps, 3)
    """
    with open(demo_file, "rb") as f:
        data = pickle.load(f)
    return jnp.array(data["obs"]), jnp.array(data["labels"])


def train_model(obs, labels, loss_type="mse", epochs=100, batch_size=32, hidden_size=128, lr=1e-4):
    """
    Train a neural network model using behavioral cloning.

    Args:
        obs: Observations array of shape (num_steps, obs_dim)
        labels: Target actions array of shape (num_steps, 3)
        loss_type: Type of loss function to use ("mse", "huber", "l1", or "combined")
        epochs: Number of training epochs
        batch_size: Batch size for training
        hidden_size: Size of hidden layers in the neural network
        lr: Learning rate

    Returns:
        Tuple of (trained model, loss history)
    """
    model = HipKneeController(
        input_size=obs.shape[1],
        hidden_size=hidden_size,
        key=jax.random.PRNGKey(42)
    )
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @jax.jit
    def loss_fn(m, o, y):
        preds = jax.vmap(m)(o)
        if loss_type == "mse":
            return jnp.mean((preds - y) ** 2)
        elif loss_type == "huber":
            return jnp.mean(optax.huber_loss(preds - y, delta=1.0))
        elif loss_type == "l1":
            return jnp.mean(jnp.abs(preds - y))
        elif loss_type == "combined":
            # Combined loss: MSE + L1 + Huber
            mse_loss = jnp.mean((preds - y) ** 2)
            l1_loss = jnp.mean(jnp.abs(preds - y))
            huber_loss = jnp.mean(optax.huber_loss(preds - y, delta=1.0))
            return (mse_loss + l1_loss + huber_loss) / 3.0
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    @jax.jit
    def step(m, st, o, y):
        g = jax.grad(loss_fn)(m, o, y)
        upd, st = optimizer.update(g, st)
        return eqx.apply_updates(m, upd), st

    n = obs.shape[0]
    loss_history = []
    for ep in range(1, epochs + 1):
        perm = jax.random.permutation(jax.random.PRNGKey(ep), n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            model, opt_state = step(model, opt_state, obs[idx], labels[idx])
        loss = float(loss_fn(model, obs, labels))
        loss_history.append(loss)
        print(f"[{loss_type.upper()}] epoch {ep:02d}  loss={loss:.4f}")

    return model, loss_history


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