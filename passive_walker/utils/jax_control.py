"""
JAX-optimized control and reward functions for high-performance computation.

This module provides JIT-compiled functions for PD control and reward calculation
that can be used for single environments or batched across multiple environments.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Union
import numpy as np

# ============================================================================
# JAX JIT-Compiled PD Control Function
# ============================================================================

@jax.jit
def pd_step(q: jnp.ndarray, qd: jnp.ndarray, q_des: jnp.ndarray, 
            kp: jnp.ndarray, kv: jnp.ndarray, 
            umin: jnp.ndarray, umax: jnp.ndarray) -> jnp.ndarray:
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
    # PD control: u = kp * (q_des - q) - kv * qd
    u = kp * (q_des - q) - kv * qd
    
    # Clamp to control limits
    return jnp.clip(u, umin, umax)

# ============================================================================
# JAX JIT-Compiled Quaternion to Euler Conversion
# ============================================================================

@jax.jit
def quat2euler_zyx_normalized(q: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Convert quaternions to ZYX Euler angles using JAX (with normalization).
    
    Args:
        q: Quaternion(s) in MuJoCo format [w, x, y, z]
            Shape can be (4,) for single quaternion or (N, 4) for batch
        eps: Small value to avoid division by zero (default: 1e-12)
        
    Returns:
        Euler angles [roll, pitch, yaw] in ZYX convention
        Shape matches input: (3,) or (N, 3)
    """
    q = jnp.asarray(q)
    
    # Normalize quaternion
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
    
    # Extract quaternion components
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Direct Euler angle formulas (no 3x3 matrix needed)
    # Roll (rotation around x-axis)
    roll = jnp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    
    # Pitch (rotation around y-axis)
    t2 = 2 * (w*y - z*x)
    t2 = jnp.clip(t2, -1.0, 1.0)  # Keep within [-1, 1] for arcsin
    pitch = jnp.arcsin(t2)
    
    # Yaw (rotation around z-axis)
    yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return jnp.stack([roll, pitch, yaw], axis=-1)

@jax.jit
def quat2euler_zyx_unnormalized(q: jnp.ndarray) -> jnp.ndarray:
    """
    Convert quaternions to ZYX Euler angles using JAX (assumes already normalized).
    
    Args:
        q: Quaternion(s) in MuJoCo format [w, x, y, z] (already normalized)
            Shape can be (4,) for single quaternion or (N, 4) for batch
        
    Returns:
        Euler angles [roll, pitch, yaw] in ZYX convention
        Shape matches input: (3,) or (N, 3)
    """
    q = jnp.asarray(q)
    
    # Extract quaternion components (assume already normalized)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Direct Euler angle formulas (no 3x3 matrix needed)
    # Roll (rotation around x-axis)
    roll = jnp.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    
    # Pitch (rotation around y-axis)
    t2 = 2 * (w*y - z*x)
    t2 = jnp.clip(t2, -1.0, 1.0)  # Keep within [-1, 1] for arcsin
    pitch = jnp.arcsin(t2)
    
    # Yaw (rotation around z-axis)
    yaw = jnp.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return jnp.stack([roll, pitch, yaw], axis=-1)

# Convenience functions
def quat2euler_zyx(q: jnp.ndarray, normalize: bool = True, eps: float = 1e-12) -> jnp.ndarray:
    """
    Convert quaternions to ZYX Euler angles using JAX for maximum performance.
    
    Args:
        q: Quaternion(s) in MuJoCo format [w, x, y, z]
            Shape can be (4,) for single quaternion or (N, 4) for batch
        normalize: Whether to normalize quaternions (default: True)
        eps: Small value to avoid division by zero (default: 1e-12)
        
    Returns:
        Euler angles [roll, pitch, yaw] in ZYX convention
        Shape matches input: (3,) or (N, 3)
    """
    if normalize:
        return quat2euler_zyx_normalized(q, eps)
    else:
        return quat2euler_zyx_unnormalized(q)

# JIT-compiled version for maximum speed (assumes quaternions are already normalized)
quat2euler_zyx_jit = quat2euler_zyx_unnormalized

# ============================================================================
# JAX JIT-Compiled Reward Function
# ============================================================================

@jax.jit
def reward_fn(dx: jnp.ndarray, pitch_abs: jnp.ndarray, ctrl_abs_sum: jnp.ndarray,
              left_z: jnp.ndarray, right_z: jnp.ndarray, torso_z: jnp.ndarray,
              upright_pitch: float, foot_clear: float,
              fall_pitch_max: float, fall_z_min: float, dx_min: float,
              vx: jnp.ndarray = None, vx_min: float = 0.005,
              c_fp: float = 1.0, c_up: float = 0.5, 
              c_fc: float = 0.2, c_ac: float = 0.1,
              pen_fall: float = 5.0, pen_stall: float = 0.2) -> jnp.ndarray:
    """
    JIT-compiled reward function with fall and stall detection.
    
    Args:
        dx: Forward progress (scalar or (N,))
        pitch_abs: Absolute pitch angle (scalar or (N,))
        ctrl_abs_sum: Sum of absolute control values (scalar or (N,))
        left_z: Left foot z position (scalar or (N,))
        right_z: Right foot z position (scalar or (N,))
        torso_z: Torso z position (scalar or (N,))
        upright_pitch: Upright threshold (scalar)
        foot_clear: Foot clearance threshold (scalar)
        fall_pitch_max: Fall detection pitch threshold (scalar)
        fall_z_min: Fall detection height threshold (scalar)
        dx_min: Stagnation detection threshold (scalar)
        vx: Forward velocity (scalar or (N,)) - optional for velocity-based stall
        vx_min: Velocity-based stall threshold (scalar)
        c_fp: Forward progress coefficient (scalar)
        c_up: Upright bonus coefficient (scalar)
        c_fc: Foot clearance coefficient (scalar)
        c_ac: Action cost coefficient (scalar)
        pen_fall: Fall penalty magnitude (scalar)
        pen_stall: Stall penalty magnitude (scalar)
        
    Returns:
        Reward value with fall and stall penalties (scalar or (N,))
    """
    # Base terms
    upright = jnp.where(pitch_abs < upright_pitch, c_up, 0.0)
    feet = jnp.where((left_z > foot_clear) & (right_z > foot_clear), c_fc, 0.0)
    base = c_fp * dx + upright + feet - c_ac * ctrl_abs_sum
    
    # Fall and stall conditions
    fell = (pitch_abs > fall_pitch_max) | (torso_z < fall_z_min)
    
    # Stall detection: use velocity if provided, otherwise use position delta
    if vx is not None:
        stall = jnp.abs(vx) < vx_min  # Velocity-based stall detection
    else:
        stall = jnp.abs(dx) < dx_min  # Position-based stall detection
    
    # Apply penalties
    return base - pen_fall * fell.astype(jnp.float32) - pen_stall * stall.astype(jnp.float32)

# ============================================================================
# Batched Functions with vmap
# ============================================================================

# Batched PD control for N environments
v_pd_step = jax.jit(jax.vmap(pd_step, in_axes=(0, 0, 0, None, None, None, None)))

# Batched reward function for N environments  
v_reward_fn = jax.jit(jax.vmap(reward_fn, in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None, 0, None, None, None, None, None, None, None)))

# ============================================================================
# Utility Functions
# ============================================================================

def get_pd_gains_jax() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get PD gains as JAX arrays."""
    kp = jnp.array([5.0, 1000.0, 1000.0], dtype=jnp.float32)  # hip, left_knee, right_knee
    kv = jnp.array([1.0, 100.0, 100.0], dtype=jnp.float32)  # Match original control.py gains
    return kp, kv

def get_control_limits_jax() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get control limits as JAX arrays."""
    umin = jnp.array([-50.0, -800.0, -800.0], dtype=jnp.float32)  # hip, left_knee, right_knee
    umax = jnp.array([50.0, 800.0, 800.0], dtype=jnp.float32)
    return umin, umax

def numpy_to_jax_single(q_np: np.ndarray, qd_np: np.ndarray, qdes_np: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert numpy arrays to JAX arrays for single environment."""
    return jnp.asarray(q_np), jnp.asarray(qd_np), jnp.asarray(qdes_np)

def numpy_to_jax_batch(q_np: np.ndarray, qd_np: np.ndarray, qdes_np: np.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert numpy arrays to JAX arrays for batched environments."""
    return jnp.asarray(q_np), jnp.asarray(qd_np), jnp.asarray(qdes_np)

# ============================================================================
# Example Usage Functions
# ============================================================================

def compute_pd_control_jax(q: np.ndarray, qd: np.ndarray, q_des: np.ndarray) -> np.ndarray:
    """
    Compute PD control using JAX for single environment.
    
    Args:
        q: Current joint positions (3,)
        qd: Current joint velocities (3,)
        q_des: Desired joint positions (3,)
        
    Returns:
        Control torques/forces (3,)
    """
    # Convert to JAX arrays
    q_jax, qd_jax, qdes_jax = numpy_to_jax_single(q, qd, q_des)
    
    # Get gains and limits
    kp, kv = get_pd_gains_jax()
    umin, umax = get_control_limits_jax()
    
    # Compute control
    u_jax = pd_step(q_jax, qd_jax, qdes_jax, kp, kv, umin, umax)
    
    # Convert back to numpy
    return np.array(u_jax)

def compute_reward_jax(dx: float, pitch_abs: float, ctrl_abs_sum: float,
                      left_z: float, right_z: float, torso_z: float,
                      upright_pitch: float = 0.2, foot_clear: float = 0.03,
                      fall_pitch_max: float = 1.0, fall_z_min: float = 0.15, dx_min: float = 1e-3,
                      vx: float = None, vx_min: float = 0.005,
                      c_fp: float = 1.0, c_up: float = 0.5, 
                      c_fc: float = 0.2, c_ac: float = 0.1,
                      pen_fall: float = 5.0, pen_stall: float = 0.2) -> float:
    """
    Compute reward using JAX for single environment with fall and stall detection.
    
    Args:
        dx: Forward progress
        pitch_abs: Absolute pitch angle
        ctrl_abs_sum: Sum of absolute control values
        left_z: Left foot z position
        right_z: Right foot z position
        torso_z: Torso z position
        upright_pitch: Upright threshold
        foot_clear: Foot clearance threshold
        fall_pitch_max: Fall detection pitch threshold
        fall_z_min: Fall detection height threshold
        dx_min: Stagnation detection threshold
        vx: Forward velocity (optional for velocity-based stall)
        vx_min: Velocity-based stall threshold
        c_fp: Forward progress coefficient
        c_up: Upright bonus coefficient
        c_fc: Foot clearance coefficient
        c_ac: Action cost coefficient
        pen_fall: Fall penalty magnitude
        pen_stall: Stall penalty magnitude
        
    Returns:
        Reward value with fall and stall penalties
    """
    # Convert to JAX arrays
    dx_jax = jnp.asarray(dx, dtype=jnp.float32)
    pitch_abs_jax = jnp.asarray(pitch_abs, dtype=jnp.float32)
    ctrl_abs_sum_jax = jnp.asarray(ctrl_abs_sum, dtype=jnp.float32)
    left_z_jax = jnp.asarray(left_z, dtype=jnp.float32)
    right_z_jax = jnp.asarray(right_z, dtype=jnp.float32)
    torso_z_jax = jnp.asarray(torso_z, dtype=jnp.float32)
    
    # Convert velocity to JAX array if provided
    vx_jax = jnp.asarray(vx, dtype=jnp.float32) if vx is not None else None
    
    # Compute reward with custom coefficients
    reward_jax = reward_fn(dx_jax, pitch_abs_jax, ctrl_abs_sum_jax,
                          left_z_jax, right_z_jax, torso_z_jax,
                          upright_pitch, foot_clear,
                          fall_pitch_max, fall_z_min, dx_min,
                          vx_jax, vx_min,
                          c_fp, c_up, c_fc, c_ac,
                          pen_fall, pen_stall)
    
    # Convert back to float
    return float(reward_jax)
