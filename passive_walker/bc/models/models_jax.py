"""
JAX Neural Network Models for BC

Defines MLP architectures using Equinox for JAX-based behavior cloning.
Supports functional programming paradigm with immutable models.
"""

from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx


class Linear(eqx.Module):
    """
    Custom linear layer that mimics PyTorch's Linear layer.
    
    Uses Xavier initialization for stable training.
    """
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_dim: int, out_dim: int, *, key):
        """
        Initialize linear layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            key: JAX random key
        """
        # Xavier initialization
        self.weight = jax.random.normal(key, (out_dim, in_dim)) * jnp.sqrt(2.0 / in_dim)
        self.bias = jnp.zeros((out_dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass: y = xW^T + b"""
        return x @ self.weight.T + self.bias


class MLP(eqx.Module):
    """
    Equinox-based MLP for BC with GELU activations and Tanh output.
    
    Architecture: Input -> [Hidden layers] -> Output
    - Configurable width and depth
    - GELU activations for smooth gradients
    - Tanh output ensures actions stay in [-1, 1] range
    - Functional programming paradigm (immutable)
    """
    
    layers: list
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 2, *, key):
        """
        Initialize MLP.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            width: Hidden layer width
            depth: Number of hidden layers
            key: JAX random key
        """
        keys = jax.random.split(key, depth + 1)
        layers = []
        last = in_dim
        
        # Hidden layers
        for i in range(depth):
            layers.append(Linear(last, width, key=keys[i]))
            last = width
        
        # Output layer
        layers.append(Linear(last, out_dim, key=keys[-1]))
        
        self.layers = layers
        self.out_dim = out_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with actions in [-1, 1] range
        """
        # Hidden layers with GELU activation
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        
        # Output layer with Tanh activation
        x = self.layers[-1](x)
        return jnp.tanh(x)


def make_model(in_dim: int, out_dim: int, width: int, depth: int, key) -> MLP:
    """
    Create a new MLP model.
    
    Args:
        in_dim: Input dimension
        out_dim: Output dimension
        width: Hidden layer width
        depth: Number of hidden layers
        key: JAX random key
        
    Returns:
        New MLP model
    """
    return MLP(in_dim, out_dim, width=width, depth=depth, key=key)


# --- Save/Load helpers (Equinox-native) ---
def save_eqx(path: str, model: eqx.Module) -> None:
    """
    Save Equinox model to file.
    
    Args:
        path: File path to save to
        model: Model to save
    """
    eqx.tree_serialise_leaves(path, model)


def load_eqx(path: str, template: eqx.Module = None) -> eqx.Module:
    """
    Load Equinox model from file.
    
    Args:
        path: File path to load from
        template: Template model with same architecture
        
    Returns:
        Loaded model
        
    Raises:
        ValueError: If template is not provided
    """
    if template is None:
        raise ValueError("Template model required for loading. Use load_eqx_with_template instead.")
    return eqx.tree_deserialise_leaves(path, template)


def load_eqx_with_template(path: str, in_dim: int, out_dim: int, width: int = 128, depth: int = 2) -> eqx.Module:
    """
    Load Equinox model with automatic template creation.
    
    Args:
        path: File path to load from
        in_dim: Input dimension
        out_dim: Output dimension
        width: Hidden layer width
        depth: Number of hidden layers
        
    Returns:
        Loaded model
    """
    # Create template model with dummy key (will be overwritten)
    key = jax.random.PRNGKey(0)
    template = MLP(in_dim, out_dim, width=width, depth=depth, key=key)
    return eqx.tree_deserialise_leaves(path, template)