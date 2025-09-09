"""
JAX MLP models using Equinox.
"""

from __future__ import annotations
from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx


class Linear(eqx.Module):
    """Custom linear layer that works like PyTorch."""
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_dim: int, out_dim: int, *, key):
        self.weight = jax.random.normal(key, (out_dim, in_dim)) * jnp.sqrt(2.0 / in_dim)
        self.bias = jnp.zeros((out_dim,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weight.T + self.bias


class MLP(eqx.Module):
    """Equinox-based MLP with GELU activations and Tanh output head."""
    
    layers: list
    out_dim: int

    def __init__(self, in_dim: int, out_dim: int, width: int = 128, depth: int = 2, *, key):
        keys = jax.random.split(key, depth + 1)
        layers = []
        last = in_dim
        for i in range(depth):
            layers.append(Linear(last, width, key=keys[i]))
            last = width
        layers.append(Linear(last, out_dim, key=keys[-1]))
        self.layers = layers
        self.out_dim = out_dim

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        x = self.layers[-1](x)
        # Actions in [-1, 1]
        return jnp.tanh(x)


def make_model(in_dim: int, out_dim: int, width: int, depth: int, key) -> MLP:
    """Create a new MLP model."""
    return MLP(in_dim, out_dim, width=width, depth=depth, key=key)


# --- Save/Load helpers (Equinox-native) ---
def save_eqx(path: str, model: eqx.Module) -> None:
    """Save Equinox model to file."""
    eqx.tree_serialise_leaves(path, model)


def load_eqx(path: str, template: eqx.Module = None) -> eqx.Module:
    """Load Equinox model from file."""
    if template is None:
        # Create a dummy template - this won't work for complex models
        # For now, we'll need to pass the template
        raise ValueError("Template model required for loading. Use load_eqx_with_template instead.")
    return eqx.tree_deserialise_leaves(path, template)


def load_eqx_with_template(path: str, in_dim: int, out_dim: int, width: int = 128, depth: int = 2) -> eqx.Module:
    """Load Equinox model with template creation."""
    import jax
    # Create template model
    key = jax.random.PRNGKey(0)  # Dummy key, will be overwritten
    template = MLP(in_dim, out_dim, width=width, depth=depth, key=key)
    return eqx.tree_deserialise_leaves(path, template)
