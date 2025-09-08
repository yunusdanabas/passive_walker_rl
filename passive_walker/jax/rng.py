"""
JAX PRNG utilities for deterministic random number generation.

Provides JAX-compatible random number generation functions that can be
efficiently vectorized and JIT-compiled for parallel environment sampling.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Union, Dict


def make_key(seed: int) -> jax.random.PRNGKey:
    """
    Create a JAX PRNG key from a seed.
    
    Args:
        seed: Integer seed value
        
    Returns:
        JAX PRNG key
    """
    return jax.random.PRNGKey(seed)


def split(key: jax.random.PRNGKey, num: int = 2) -> Tuple[jax.random.PRNGKey, ...]:
    """
    Split a PRNG key into multiple keys.
    
    Args:
        key: Input PRNG key
        num: Number of keys to split into
        
    Returns:
        Tuple of PRNG keys
    """
    return jax.random.split(key, num)


def fold_in(key: jax.random.PRNGKey, data: int) -> jax.random.PRNGKey:
    """
    Fold additional data into a PRNG key.
    
    Args:
        key: Input PRNG key
        data: Integer data to fold in
        
    Returns:
        New PRNG key
    """
    return jax.random.fold_in(key, data)


def uniform(key: jax.random.PRNGKey, shape: Tuple[int, ...], 
           minval: float = 0.0, maxval: float = 1.0) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    """
    Generate uniform random values.
    
    Args:
        key: PRNG key
        shape: Output shape
        minval: Minimum value (inclusive)
        maxval: Maximum value (exclusive)
        
    Returns:
        (new_key, random_values)
    """
    new_key, subkey = jax.random.split(key)
    values = jax.random.uniform(subkey, shape, minval=minval, maxval=maxval)
    return new_key, values


def normal(key: jax.random.PRNGKey, shape: Tuple[int, ...], 
          mean: float = 0.0, std: float = 1.0) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    """
    Generate normal random values.
    
    Args:
        key: PRNG key
        shape: Output shape
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        
    Returns:
        (new_key, random_values)
    """
    new_key, subkey = jax.random.split(key)
    values = jax.random.normal(subkey, shape) * std + mean
    return new_key, values


def choice(key: jax.random.PRNGKey, a: Union[int, jnp.ndarray], 
          shape: Tuple[int, ...] = (), replace: bool = True, 
          p: jnp.ndarray = None) -> Tuple[jax.random.PRNGKey, jnp.ndarray]:
    """
    Generate random choices from a given array.
    
    Args:
        key: PRNG key
        a: Array to choose from or size of array
        shape: Output shape
        replace: Whether to sample with replacement
        p: Probabilities for each element (optional)
        
    Returns:
        (new_key, random_choices)
    """
    new_key, subkey = jax.random.split(key)
    choices = jax.random.choice(subkey, a, shape=shape, replace=replace, p=p)
    return new_key, choices


def domain_randomize_physics(key: jax.random.PRNGKey, 
                           ramp_deg_min: float, ramp_deg_max: float,
                           friction_min: float, friction_max: float,
                           mass_jitter: float, base_mass: float) -> Tuple[jax.random.PRNGKey, Dict[str, jnp.ndarray]]:
    """
    Generate domain randomization parameters for physics.
    
    Args:
        key: PRNG key
        ramp_deg_min: Minimum ramp angle (degrees)
        ramp_deg_max: Maximum ramp angle (degrees)
        friction_min: Minimum friction coefficient
        friction_max: Maximum friction coefficient
        mass_jitter: Mass jitter factor (0.0 = no jitter, 0.1 = ±10%)
        base_mass: Base mass value
        
    Returns:
        (new_key, physics_params) where physics_params contains:
        - gravity: [gx, gy, gz] gravity vector
        - friction: friction coefficient
        - mass: jittered mass value
    """
    key, subkey1, subkey2 = split(key, 3)
    
    # Ramp angle
    key, ramp_deg = uniform(key, (), ramp_deg_min, ramp_deg_max)
    ramp_rad = jnp.deg2rad(ramp_deg)
    gravity = jnp.array([
        9.81 * jnp.sin(ramp_rad),
        0.0,
        -9.81 * jnp.cos(ramp_rad)
    ])
    
    # Friction
    key, friction = uniform(key, (), friction_min, friction_max)
    
    # Mass jitter
    key, mass_scale = uniform(key, (), 1.0 - mass_jitter, 1.0 + mass_jitter)
    mass = base_mass * mass_scale
    
    return key, {
        "gravity": gravity,
        "friction": friction,
        "mass": mass
    }


# JIT-compiled versions for performance
make_key_jit = jax.jit(make_key)
split_jit = jax.jit(split)
fold_in_jit = jax.jit(fold_in)
uniform_jit = jax.jit(uniform)
normal_jit = jax.jit(normal)
choice_jit = jax.jit(choice)
domain_randomize_physics_jit = jax.jit(domain_randomize_physics)