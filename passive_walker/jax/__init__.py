"""
JAX utilities for passive walker RL.

This module provides JAX-accelerated versions of core functionality
for vectorized environments and faster computation.
"""

from .controller_jax import pd_step, pd_step_vmap, pd_step_broadcast, quat2euler_zyx, v_quat2euler_zyx
from .reward_jax import minimal_reward, research_reward, aggressive_reward, get_reward_function, make_batched_reward_fn
from .rng import make_key, split, fold_in, uniform, normal, choice, domain_randomize_physics

__all__ = [
    # Controller functions
    "pd_step",
    "pd_step_vmap", 
    "pd_step_broadcast",
    "quat2euler_zyx",
    "v_quat2euler_zyx",
    # Reward functions
    "minimal_reward",
    "research_reward",
    "aggressive_reward",
    "get_reward_function",
    "make_batched_reward_fn",
    # RNG functions
    "make_key",
    "split",
    "fold_in",
    "uniform",
    "normal",
    "choice",
    "domain_randomize_physics"
]
