"""
JAX Controller for Passive Walker

JIT-compiled PD control and quaternion utilities for high-performance
computation. Mirrors controller.py parameters exactly.

NOTE: This is the canonical JAX controller. See also passive_walker/jax/controller_jax.py
for vectorized versions. Keep implementations in sync to avoid divergence.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

# =====================
# Constants (mirror controller.py)
# =====================
KP = jnp.array([5.0, 1000.0, 1000.0], dtype=jnp.float32)
KD = jnp.array([1.0, 100.0, 100.0], dtype=jnp.float32)
U_MIN = jnp.array([-50.0, -800.0, -800.0], dtype=jnp.float32)
U_MAX = jnp.array([50.0, 800.0, 800.0], dtype=jnp.float32)


# =====================
# JIT-Compiled Functions
# =====================
@jax.jit
def pd_step(q: jnp.ndarray, qd: jnp.ndarray, q_des: jnp.ndarray,
            kp: jnp.ndarray = KP, kv: jnp.ndarray = KD,
            umin: jnp.ndarray = U_MIN, umax: jnp.ndarray = U_MAX) -> jnp.ndarray:
    """
    JIT-compiled PD control step.
    
    Args:
        q: Current joint positions (3,) or (N, 3)
        qd: Current joint velocities (3,) or (N, 3)
        q_des: Desired joint positions (3,) or (N, 3)
        kp: Proportional gains (3,)
        kv: Derivative gains (3,)
        umin: Minimum control limits (3,)
        umax: Maximum control limits (3,)
        
    Returns:
        Control torques/forces (3,) or (N, 3)
    """
    u = kp * (q_des - q) - kv * qd
    return jnp.clip(u, umin, umax)


@jax.jit
def quat2euler_zyx_normalized(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Convert quaternions to ZYX Euler angles with normalization.
    
    Args:
        q: Quaternion(s) in MuJoCo format [w, x, y, z]
            Shape: (4,) for single or (N, 4) for batch
        eps: Small value to avoid division by zero
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
        Shape: (3,) for single or (N, 3) for batch
    """
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    roll = jnp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    t2 = jnp.clip(2*(w*y - z*x), -1.0, 1.0)
    pitch = jnp.arcsin(t2)
    yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return jnp.stack([roll, pitch, yaw], axis=-1)


# =====================
# Batched Functions
# =====================
# Batched PD control for N environments
v_pd_step = jax.jit(jax.vmap(pd_step, in_axes=(0, 0, 0, None, None, None, None)))

# Batched quaternion to Euler conversion
v_quat2euler_zyx = jax.jit(jax.vmap(quat2euler_zyx_normalized, in_axes=(0, None)))


# =====================
# Convenience Functions
# =====================
def quat2euler_zyx(q: jnp.ndarray, normalize: bool = True, eps: float = 1e-12) -> jnp.ndarray:
    """
    Convert quaternions to ZYX Euler angles.
    
    Args:
        q: Quaternion(s) in MuJoCo format [w, x, y, z]
        normalize: Whether to normalize quaternions (default: True)
        eps: Small value to avoid division by zero
        
    Returns:
        Euler angles [roll, pitch, yaw] in radians
    """
    if normalize:
        return quat2euler_zyx_normalized(q, eps)
    else:
        # Assume already normalized for slight performance gain
        return quat2euler_zyx_normalized(q, 0.0)