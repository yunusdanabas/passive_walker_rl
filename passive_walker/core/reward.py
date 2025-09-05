"""
Reward API:
- get_reward_fn(preset) -> callable
- minimal/default/aggressive presets from YAML
"""

import numpy as np


def get_reward_fn(preset: str):
    """
    Returns a callable: fn(dx, pitch_abs, u_abs_sum, vx, lk_q, rk_q, left_foot_z, right_foot_z, torso_z) -> float
    """
    if preset == "minimal":
        return _minimal_reward
    elif preset == "default":
        return _default_reward
    elif preset == "aggressive":
        return _aggressive_reward
    else:
        raise ValueError(f"Unknown reward preset: {preset}")


def _minimal_reward(
    *, dx, pitch_abs, u_abs_sum, vx, lk_q, rk_q, left_foot_z, right_foot_z, torso_z
) -> float:
    """Minimal reward: forward progress only."""
    return float(dx)


def _default_reward(
    *, dx, pitch_abs, u_abs_sum, vx, lk_q, rk_q, left_foot_z, right_foot_z, torso_z
) -> float:
    """Default reward: forward progress + basic shaping."""
    # Forward progress
    reward = float(dx)

    # Upright bonus
    if pitch_abs < 0.25:
        reward += 0.5

    # Small action penalty
    reward -= 3e-4 * u_abs_sum

    return reward


def _aggressive_reward(
    *, dx, pitch_abs, u_abs_sum, vx, lk_q, rk_q, left_foot_z, right_foot_z, torso_z
) -> float:
    """Aggressive reward: heavily shaped for RL training."""
    # Forward progress
    reward = 2.0 * float(dx)

    # Upright bonus
    if pitch_abs < 0.25:
        reward += 1.0

    # Action penalty
    reward -= 1e-3 * u_abs_sum

    # Velocity tracking
    vx_target = 0.9
    vx_error = abs(vx - vx_target)
    reward += 0.5 * np.exp(-vx_error / 0.2)

    # Symmetry bonus
    knee_diff = abs(lk_q - rk_q)
    reward += 0.1 * np.exp(-knee_diff / 0.35)

    # Foot clearance
    foot_clear_target = 0.03
    lk_clear = max(0, left_foot_z - foot_clear_target)
    rk_clear = max(0, right_foot_z - foot_clear_target)
    reward += 0.1 * (lk_clear + rk_clear)

    return reward
