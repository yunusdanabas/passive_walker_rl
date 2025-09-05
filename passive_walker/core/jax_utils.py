"""
JAX utilities for fast, JIT-compiled primitives.

- pd_step(): Single and batched PD control
- quat2euler_zyx(): Quaternion to Euler conversion
- make_batched_reward_fn(): Vectorize existing reward functions
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Callable


# ---------- PD CONTROL ----------
@jax.jit
def pd_step(
    q: jnp.ndarray,
    qd: jnp.ndarray,
    q_des: jnp.ndarray,
    kp: jnp.ndarray,
    kv: jnp.ndarray,
    umin: jnp.ndarray,
    umax: jnp.ndarray,
) -> jnp.ndarray:
    """Single-env PD step, shape (3,) each. Returns clipped torques/forces."""
    u = kp * (q_des - q) - kv * qd
    return jnp.clip(u, umin, umax)


# Batched PD: (N,3) everywhere; gains/limits broadcast
v_pd_step = jax.jit(jax.vmap(pd_step, in_axes=(0, 0, 0, None, None, None, None)))


# ---------- QUAT -> EULER (ZYX) ----------
@jax.jit
def quat2euler_zyx(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Convert MuJoCo-format quat [w,x,y,z] -> [roll, pitch, yaw] (ZYX).
    Accepts shape (4,) or (N,4); returns (3,) or (N,3) accordingly.
    """
    q = jnp.asarray(q)
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True) + eps
    w, x, y, z = (q / norm)[..., 0], (q / norm)[..., 1], (q / norm)[..., 2], (q / norm)[..., 3]

    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    t2 = jnp.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = jnp.arcsin(t2)
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return jnp.stack([roll, pitch, yaw], axis=-1)


# For clarity (and JIT caching)
v_quat2euler_zyx = jax.jit(jax.vmap(quat2euler_zyx, in_axes=0))


# ---------- BATCHED REWARD THIN WRAPPER ----------
def make_batched_reward_fn(
    single_reward_fn: Callable[[dict], tuple[float, dict]],
) -> Callable[[dict], tuple[jnp.ndarray, dict]]:
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
