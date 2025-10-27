"""
Smooth, RL-friendly reward functions optimized with JAX.
Designed for the Research environment to provide stable learning signals.
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RewCfg:
    """Reward configuration for RL training environment."""
    # Core terms
    c_fp: float = 1.0                    # Forward progress coefficient
    c_up: float = 0.5                    # Upright bonus coefficient
    upright_pitch_max: float = 0.25      # Max pitch for upright bonus (rad)
    c_ac: float = 3e-4                   # Action cost coefficient
    
    # Optional terms
    c_vt: float = 0.25                   # Velocity tracking coefficient
    vx_star: float = 0.8                 # Target velocity (m/s)
    sigma_v: float = 0.25                # Velocity tracking width
    
    c_sym: float = 0.05                  # Symmetry coefficient
    sigma_sym: float = 0.4               # Symmetry width
    
    c_fc: float = 0.05                   # Foot clearance coefficient
    foot_clear_target: float = 0.03      # Target foot clearance (m)
    
    # Terminal conditions
    pen_fall: float = 5.0                # Fall penalty
    fall_pitch_max: float = 1.0          # Max pitch before fall (rad)
    fall_z_min: float = 0.15             # Min torso height before fall (m)
    
    # Reward clipping
    clip_low: float = -5.0               # Lower reward bound
    clip_high: float = 5.0               # Upper reward bound


def soft_hinge(x: jnp.ndarray) -> jnp.ndarray:
    """Smooth approximation of max(0, x) using softplus."""
    return jnp.log1p(jnp.exp(x))


@jax.jit
def compute_smooth_reward(
    dx: jnp.ndarray,
    pitch_abs: jnp.ndarray,
    u_abs_sum: jnp.ndarray,
    vx: jnp.ndarray,
    lk_q: jnp.ndarray,
    rk_q: jnp.ndarray,
    left_foot_z: jnp.ndarray,
    right_foot_z: jnp.ndarray,
    torso_z: jnp.ndarray,
    cfg: RewCfg
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute smooth, RL-friendly reward using JAX.
    
    Args:
        dx: Forward progress (m)
        pitch_abs: Absolute pitch angle (rad)
        u_abs_sum: Sum of absolute control values
        vx: Forward velocity (m/s)
        lk_q: Left knee position (m)
        rk_q: Right knee position (m)
        left_foot_z: Left foot height (m)
        right_foot_z: Right foot height (m)
        torso_z: Torso height (m)
        cfg: Reward configuration
        
    Returns:
        Tuple of (reward, fell) where fell is boolean
    """
    # 1. Forward progress (dense)
    forward = cfg.c_fp * dx
    
    # 2. Smooth upright bonus (no hard thresholds)
    upright_ratio = pitch_abs / cfg.upright_pitch_max
    upright = cfg.c_up * jnp.clip(1.0 - upright_ratio**2, 0.0, 1.0)
    
    # 3. Action cost (L1 norm)
    action_cost = cfg.c_ac * u_abs_sum
    
    # 4. Velocity tracking (encourage target speed)
    vel_diff = (vx - cfg.vx_star) / cfg.sigma_v
    vel_tracking = cfg.c_vt * jnp.exp(-vel_diff**2 / 2.0)
    
    # 5. Symmetry (penalize left-right bias)
    knee_diff = (lk_q - rk_q) / cfg.sigma_sym
    symmetry = cfg.c_sym * jnp.exp(-knee_diff**2 / 2.0)
    
    # 6. Soft foot clearance (smooth instead of hard threshold)
    left_clear = soft_hinge(left_foot_z - cfg.foot_clear_target)
    right_clear = soft_hinge(right_foot_z - cfg.foot_clear_target)
    foot_clearance = cfg.c_fc * 0.5 * (left_clear + right_clear)
    
    # 7. Fall detection and penalty
    fell = (pitch_abs > cfg.fall_pitch_max) | (torso_z < cfg.fall_z_min)
    fall_penalty = jnp.where(fell, cfg.pen_fall, 0.0)
    
    # 8. Total reward (smooth, dense, clipped)
    raw_reward = (forward + upright + vel_tracking + 
                 symmetry + foot_clearance - action_cost - fall_penalty)
    reward = jnp.clip(raw_reward, cfg.clip_low, cfg.clip_high)
    
    return reward, fell


@jax.jit
def compute_simple_reward(
    dx: jnp.ndarray,
    pitch_abs: jnp.ndarray,
    u_abs_sum: jnp.ndarray,
    torso_z: jnp.ndarray,
    cfg: RewCfg
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute simplified reward for faster evaluation.
    Only includes core terms: forward progress, upright bonus, action cost, fall penalty.
    
    Args:
        dx: Forward progress (m)
        pitch_abs: Absolute pitch angle (rad)
        u_abs_sum: Sum of absolute control values
        torso_z: Torso height (m)
        cfg: Reward configuration
        
    Returns:
        Tuple of (reward, fell) where fell is boolean
    """
    # 1. Forward progress (dense)
    forward = cfg.c_fp * dx
    
    # 2. Smooth upright bonus (no hard thresholds)
    upright_ratio = pitch_abs / cfg.upright_pitch_max
    upright = cfg.c_up * jnp.clip(1.0 - upright_ratio**2, 0.0, 1.0)
    
    # 3. Action cost (L1 norm)
    action_cost = cfg.c_ac * u_abs_sum
    
    # 4. Fall detection and penalty
    fell = (pitch_abs > cfg.fall_pitch_max) | (torso_z < cfg.fall_z_min)
    fall_penalty = jnp.where(fell, cfg.pen_fall, 0.0)
    
    # 5. Total reward (smooth, dense, clipped)
    raw_reward = forward + upright - action_cost - fall_penalty
    reward = jnp.clip(raw_reward, cfg.clip_low, cfg.clip_high)
    
    return reward, fell


# Vectorized versions for batch processing
v_compute_smooth_reward = jax.vmap(compute_smooth_reward, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None))
v_compute_simple_reward = jax.vmap(compute_simple_reward, in_axes=(0, 0, 0, 0, None))


def get_default_reward_config() -> RewCfg:
    """Get default reward configuration for RL training."""
    return RewCfg()


def get_minimal_reward_config() -> RewCfg:
    """Get minimal reward configuration for fast evaluation."""
    return RewCfg(
        c_vt=0.0,      # Disable velocity tracking
        c_sym=0.0,     # Disable symmetry
        c_fc=0.0       # Disable foot clearance
    )


def get_aggressive_reward_config() -> RewCfg:
    """Get aggressive reward configuration for challenging training."""
    return RewCfg(
        c_fp=2.0,              # Higher forward progress reward
        c_up=1.0,              # Higher upright bonus
        c_ac=1e-3,             # Higher action cost
        c_vt=0.5,              # Strong velocity tracking
        c_sym=0.1,             # Strong symmetry
        c_fc=0.1,              # Strong foot clearance
        pen_fall=10.0,         # Higher fall penalty
        clip_low=-10.0,        # Wider reward range
        clip_high=10.0
    )


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    
    # Test reward function
    cfg = get_default_reward_config()
    
    # Sample inputs
    dx = jnp.array(0.1)
    pitch_abs = jnp.array(0.1)
    u_abs_sum = jnp.array(0.5)
    vx = jnp.array(0.8)
    lk_q = jnp.array(0.0)
    rk_q = jnp.array(0.0)
    left_foot_z = jnp.array(0.05)
    right_foot_z = jnp.array(0.05)
    torso_z = jnp.array(0.3)
    
    # Compute reward
    reward, fell = compute_smooth_reward(
        dx, pitch_abs, u_abs_sum, vx, lk_q, rk_q,
        left_foot_z, right_foot_z, torso_z, cfg
    )
    
    print(f"Reward: {reward:.3f}")
    print(f"Fell: {fell}")
    
    # Test batch processing
    batch_size = 10
    dx_batch = jnp.array([0.1] * batch_size)
    pitch_abs_batch = jnp.array([0.1] * batch_size)
    u_abs_sum_batch = jnp.array([0.5] * batch_size)
    vx_batch = jnp.array([0.8] * batch_size)
    lk_q_batch = jnp.array([0.0] * batch_size)
    rk_q_batch = jnp.array([0.0] * batch_size)
    left_foot_z_batch = jnp.array([0.05] * batch_size)
    right_foot_z_batch = jnp.array([0.05] * batch_size)
    torso_z_batch = jnp.array([0.3] * batch_size)
    
    rewards, fells = v_compute_smooth_reward(
        dx_batch, pitch_abs_batch, u_abs_sum_batch, vx_batch,
        lk_q_batch, rk_q_batch, left_foot_z_batch, right_foot_z_batch,
        torso_z_batch, cfg
    )
    
    print(f"Batch rewards: {rewards}")
    print(f"Batch fells: {fells}")
