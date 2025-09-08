"""
JAX-based reward functions for vectorized reward computation.

Provides vmap-ready reward functions that match the Python reward presets
for efficient parallel reward computation during PPO training.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple


def minimal_reward(dx: jnp.ndarray, cfg: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Minimal reward function (JAX version).
    
    Args:
        dx: Forward displacement (...,)
        cfg: Reward configuration dict
        
    Returns:
        reward: Scalar reward (...,)
        info: Dict with "fell" boolean (...,)
    """
    # Minimal reward is just forward progress
    reward = dx
    fell = jnp.zeros_like(dx, dtype=bool)
    return reward, {"fell": fell}


def research_reward(dx: jnp.ndarray, pitch_abs: jnp.ndarray, u_abs_sum: jnp.ndarray,
                   torso_z: jnp.ndarray, vx: jnp.ndarray, lk_q: jnp.ndarray, rk_q: jnp.ndarray,
                   left_foot_z: jnp.ndarray, right_foot_z: jnp.ndarray,
                   cfg: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Research reward function (JAX version) matching the Python preset.
    
    Args:
        dx: Forward displacement (...,)
        pitch_abs: Absolute pitch angle (...,)
        u_abs_sum: Sum of absolute control torques (...,)
        torso_z: Torso height (...,)
        vx: Forward velocity (...,)
        lk_q: Left knee position (...,)
        rk_q: Right knee position (...,)
        left_foot_z: Left foot height (...,)
        right_foot_z: Right foot height (...,)
        cfg: Reward configuration dict
        
    Returns:
        reward: Scalar reward (...,)
        info: Dict with "fell" boolean (...,)
    """
    # Extract config parameters
    c_fp = cfg["c_fp"]
    c_up = cfg["c_up"]
    upright_pitch_max = cfg["upright_pitch_max"]
    c_ac = cfg["c_ac"]
    c_vt = cfg.get("c_vt", 0.0)
    vx_star = cfg.get("vx_star", 0.8)
    sigma_v = cfg.get("sigma_v", 0.25)
    c_sym = cfg.get("c_sym", 0.0)
    sigma_sym = cfg.get("sigma_sym", 0.4)
    c_fc = cfg.get("c_fc", 0.0)
    foot_clear_target = cfg.get("foot_clear_target", 0.03)
    pen_fall = cfg["pen_fall"]
    fall_pitch_max = cfg["fall_pitch_max"]
    fall_z_min = cfg["fall_z_min"]
    clip_low = cfg["clip_low"]
    clip_high = cfg["clip_high"]

    # Forward progress reward
    forward = c_fp * dx
    
    # Upright reward (quadratic penalty)
    upright_ratio = pitch_abs / upright_pitch_max
    upright = c_up * jnp.maximum(0.0, 1.0 - upright_ratio**2)
    
    # Action cost (penalty for high control effort)
    act_cost = c_ac * u_abs_sum
    
    # Velocity target reward (Gaussian around target speed)
    vel = c_vt * jnp.exp(-0.5 * ((vx - vx_star) / sigma_v) ** 2)
    
    # Symmetry reward (penalize knee asymmetry)
    knee_diff = (lk_q - rk_q) / sigma_sym
    sym = c_sym * jnp.exp(-0.5 * knee_diff**2)
    
    # Foot clearance reward (soft hinge via log1p(exp(.)))
    left_clear = jnp.log1p(jnp.exp(left_foot_z - foot_clear_target))
    right_clear = jnp.log1p(jnp.exp(right_foot_z - foot_clear_target))
    foot_clear = c_fc * 0.5 * (left_clear + right_clear)
    
    # Fall detection
    fell = (pitch_abs > fall_pitch_max) | (torso_z < fall_z_min)
    
    # Total reward
    total = forward + upright + vel + sym + foot_clear - act_cost - pen_fall * fell
    reward = jnp.clip(total, clip_low, clip_high)
    
    return reward, {"fell": fell}


def aggressive_reward(dx: jnp.ndarray, pitch_abs: jnp.ndarray, u_abs_sum: jnp.ndarray,
                     torso_z: jnp.ndarray, vx: jnp.ndarray, lk_q: jnp.ndarray, rk_q: jnp.ndarray,
                     left_foot_z: jnp.ndarray, right_foot_z: jnp.ndarray,
                     cfg: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Aggressive reward function (JAX version) with higher penalties and rewards.
    
    Same structure as research_reward but with different coefficients.
    """
    return research_reward(dx, pitch_abs, u_abs_sum, torso_z, vx, lk_q, rk_q,
                          left_foot_z, right_foot_z, cfg)


# Vectorized versions for batch processing
minimal_reward_vmap = jax.vmap(minimal_reward, in_axes=(0, None))
research_reward_vmap = jax.vmap(research_reward, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))
aggressive_reward_vmap = jax.vmap(aggressive_reward, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))

# JIT-compiled versions for performance
minimal_reward_jit = jax.jit(minimal_reward)
research_reward_jit = jax.jit(research_reward)
aggressive_reward_jit = jax.jit(aggressive_reward)

minimal_reward_vmap_jit = jax.jit(minimal_reward_vmap)
research_reward_vmap_jit = jax.jit(research_reward_vmap)
aggressive_reward_vmap_jit = jax.jit(aggressive_reward_vmap)


def get_reward_function(preset: str):
    """
    Get the appropriate JAX reward function by preset name.
    
    Args:
        preset: "minimal", "default", or "aggressive"
        
    Returns:
        JAX reward function
    """
    if preset == "minimal":
        return minimal_reward
    elif preset == "default":
        return research_reward
    elif preset == "aggressive":
        return aggressive_reward
    else:
        raise ValueError(f"Unknown reward preset: {preset}")


def get_reward_function_vmap(preset: str):
    """
    Get the appropriate vectorized JAX reward function by preset name.
    
    Args:
        preset: "minimal", "default", or "aggressive"
        
    Returns:
        Vectorized JAX reward function
    """
    if preset == "minimal":
        return minimal_reward_vmap
    elif preset == "default":
        return research_reward_vmap
    elif preset == "aggressive":
        return aggressive_reward_vmap
    else:
        raise ValueError(f"Unknown reward preset: {preset}")


# ---------- BATCHED REWARD WRAPPER ----------
def make_batched_reward_fn(
    single_reward_fn: callable,
) -> callable:
    """
    Wraps your existing scalar reward_fn(signals)->(reward, extras)
    into a batched version expecting each value in `signals` as jnp arrays
    with leading batch dim. Extras are aggregated per-key.
    """
    def batched(signals_batched: dict) -> tuple[jnp.ndarray, dict]:
        # Get batch size from first array-like signal
        batch_size = None
        for v in signals_batched.values():
            if isinstance(v, jnp.ndarray) and v.ndim >= 1:
                batch_size = v.shape[0]
                break

        if batch_size is None:
            # No arrays, just call single function
            r, extras = single_reward_fn(signals_batched)
            return jnp.array([r]), extras

        # For now, use a simple loop approach since JAX vmap with dicts is complex
        rewards = []
        extras_list = []

        for i in range(batch_size):
            # Extract single item from batched signals
            item = {}
            for k, v in signals_batched.items():
                if isinstance(v, jnp.ndarray) and v.ndim >= 1:
                    item[k] = v[i]
                else:
                    item[k] = v

            # Call single reward function
            r, extras = single_reward_fn(item)
            rewards.append(r)
            extras_list.append(extras)

        rewards = jnp.array(rewards, dtype=jnp.float32)

        # Reduce extras across batch if they're numeric; else return first item
        def reduce_extras(extras_list):
            if not extras_list:
                return {}

            # Get all keys from first item
            keys = list(extras_list[0].keys())
            reduced = {}

            for key in keys:
                values = [extras[key] for extras in extras_list]
                if all(isinstance(v, (int, float)) for v in values):
                    # Numeric values - take mean
                    reduced[key] = sum(values) / len(values)
                else:
                    # Non-numeric values - return first item
                    reduced[key] = values[0]

            return reduced

        extras_reduced = reduce_extras(extras_list)
        return rewards, extras_reduced

    return batched  # Don't JIT this since it has Python loops