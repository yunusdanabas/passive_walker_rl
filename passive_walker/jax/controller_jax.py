"""
JAX-based PD controller for vectorized control computations.

Provides vmap-ready PD control functions that can be efficiently batched
for vectorized environment steps and PPO training.
"""

import jax
import jax.numpy as jnp


def pd_step(q: jnp.ndarray, qd: jnp.ndarray, q_des: jnp.ndarray, 
            kp: jnp.ndarray, kv: jnp.ndarray, 
            umin: jnp.ndarray, umax: jnp.ndarray) -> jnp.ndarray:
    """
    Elementwise PD control with clipping.
    
    Args:
        q: Current joint positions (..., 3)
        qd: Current joint velocities (..., 3) 
        q_des: Desired joint positions (..., 3)
        kp: Proportional gains (..., 3)
        kv: Derivative gains (..., 3)
        umin: Minimum control limits (..., 3)
        umax: Maximum control limits (..., 3)
        
    Returns:
        Control torques (..., 3)
    """
    u = kp * (q_des - q) - kv * qd
    return jnp.clip(u, umin, umax)


def pd_step_vmap(q: jnp.ndarray, qd: jnp.ndarray, q_des: jnp.ndarray,
                 kp: jnp.ndarray, kv: jnp.ndarray,
                 umin: jnp.ndarray, umax: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorized PD control for batch processing.
    
    Batches over the leading axis (axis 0) for efficient parallel computation.
    All inputs should have shape (batch_size, 3).
    
    Args:
        q: Current joint positions (B, 3)
        qd: Current joint velocities (B, 3)
        q_des: Desired joint positions (B, 3)
        kp: Proportional gains (B, 3) or (3,) for shared gains
        kv: Derivative gains (B, 3) or (3,) for shared gains
        umin: Minimum control limits (B, 3) or (3,) for shared limits
        umax: Maximum control limits (B, 3) or (3,) for shared limits
        
    Returns:
        Control torques (B, 3)
    """
    return jax.vmap(pd_step, in_axes=(0, 0, 0, 0, 0, 0, 0))(
        q, qd, q_des, kp, kv, umin, umax
    )


def pd_step_broadcast(q: jnp.ndarray, qd: jnp.ndarray, q_des: jnp.ndarray,
                     kp: jnp.ndarray, kv: jnp.ndarray,
                     umin: jnp.ndarray, umax: jnp.ndarray) -> jnp.ndarray:
    """
    PD control with broadcasting for shared gains/limits.
    
    Useful when all environments share the same PD parameters.
    Gains and limits can be shape (3,) and will broadcast to (B, 3).
    
    Args:
        q: Current joint positions (B, 3)
        qd: Current joint velocities (B, 3)
        q_des: Desired joint positions (B, 3)
        kp: Proportional gains (3,) - shared across batch
        kv: Derivative gains (3,) - shared across batch
        umin: Minimum control limits (3,) - shared across batch
        umax: Maximum control limits (3,) - shared across batch
        
    Returns:
        Control torques (B, 3)
    """
    return jax.vmap(pd_step, in_axes=(0, 0, 0, None, None, None, None))(
        q, qd, q_des, kp, kv, umin, umax
    )


# JIT-compiled versions for performance
pd_step_jit = jax.jit(pd_step)
pd_step_vmap_jit = jax.jit(pd_step_vmap)
pd_step_broadcast_jit = jax.jit(pd_step_broadcast)


# ---------- QUATERNION UTILITIES ----------
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


# Batched quaternion conversion
v_quat2euler_zyx = jax.jit(jax.vmap(quat2euler_zyx, in_axes=0))