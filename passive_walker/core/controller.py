"""
Passive Walker Controller

PD control and FSM logic for bipedal walking. Matches legacy implementation
with quaternion-based leg angle detection and proper state transitions.
"""
from __future__ import annotations
import numpy as np

# =====================
# Physical Parameters
# =====================
# Joint ranges: hip (rad), knees (m sliders)
JOINT_MIN = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
JOINT_MAX = np.array([+0.5, +0.5, +0.5], dtype=np.float32)

# PD gains: [hip, left_knee, right_knee]
KP = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)
KD = np.array([1.0, 100.0, 100.0], dtype=np.float32)

# Control limits: [hip_torque, left_knee_force, right_knee_force]
U_MIN = np.array([-50.0, -800.0, -800.0], dtype=np.float32)
U_MAX = np.array([+50.0, +800.0, +800.0], dtype=np.float32)

# =====================
# FSM Parameters
# =====================
CONTACT_Z = 0.06           # Foot contact height threshold (m)
KNEE_RELEASE = 0.10        # Leg forward progress threshold for knee release (rad)

# Target positions
HIP_SWING_POS = +0.5       # Hip forward target (rad)
HIP_SWING_NEG = -0.5       # Hip backward target (rad)
KNEE_STANCE = 0.0          # Knee stance position (m)
KNEE_RETRACT = -0.25       # Knee retract position (m)

# Slew rate limiting (disabled - FSM requires instant transitions for balance)
HIP_SLEW = None            # rad/s - disabled: any smoothing breaks FSM balance
KNEE_SLEW = None           # m/s - disabled: any smoothing breaks FSM balance


# =====================
# Utility Functions
# =====================
def _clip(v, lo, hi):
    """Clamp value between bounds."""
    return np.minimum(np.maximum(v, lo), hi)


def _quat2euler_zyx_np(q: np.ndarray) -> tuple[float, float, float]:
    """Convert MuJoCo quaternion [w,x,y,z] to (roll, pitch, yaw) in radians."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    t2 = 2*(w*y - z*x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return float(roll), float(pitch), float(yaw)


def _slew(x: float, target: float, rate: float, dt: float) -> float:
    """Limit change per step to rate*dt. Returns target if rate is None."""
    if rate is None:
        return target
    max_delta = rate * dt
    delta = target - x
    if delta > max_delta:
        return x + max_delta
    if delta < -max_delta:
        return x - max_delta
    return target


# =====================
# PD Controller
# =====================
class PDController:
    """PD controller with per-joint gains and action denormalization.
    
    Performance Notes:
    - NumPy backend: Fastest for single-environment use cases (default)
    - JAX backend: Better for vectorized/batched operations (opt-in)
    - JAX overhead: ~15-20x slower for single env due to JIT compilation + array conversion
    - JAX benefits: Shine with 32+ parallel environments or complex computations
    """
    
    def __init__(self, use_jax: bool = False):
        self.kp = KP.copy()
        self.kd = KD.copy()
        self.umin = U_MIN.copy()
        self.umax = U_MAX.copy()
        self.jmin = JOINT_MIN.copy()
        self.jmax = JOINT_MAX.copy()
        
        # Backend selection: NumPy (fast) vs JAX (vectorized)
        self._use_jax = False
        self.backend_name = "numpy"  # Default backend name
        if use_jax:
            try:
                import jax.numpy as jnp
                from passive_walker.core.controller_jax import pd_step
                
                # Store JAX references for fast path
                self._jnp = jnp
                self._pd_step = pd_step  # Pre-JIT compiled function
                
                # Convert PD parameters to JAX arrays (compile once, use many)
                self._kp_j   = jnp.asarray(self.kp,   dtype=jnp.float32)
                self._kd_j   = jnp.asarray(self.kd,   dtype=jnp.float32)
                self._umin_j = jnp.asarray(self.umin, dtype=jnp.float32)
                self._umax_j = jnp.asarray(self.umax, dtype=jnp.float32)
                
                # Warmup compilation to avoid first-step JIT overhead
                # This moves compilation cost to initialization, not runtime
                q_warmup = jnp.zeros((3,), dtype=jnp.float32)
                _ = self._pd_step(q_warmup, q_warmup, q_warmup, 
                                self._kp_j, self._kd_j, self._umin_j, self._umax_j).block_until_ready()
                
                self._use_jax = True
                self.backend_name = "jax"
            except Exception:
                # JAX not available or failed to import; fallback to NumPy
                self._use_jax = False
                if use_jax:
                    # One-time notice so users know why it fell back
                    print("[PassiveWalker] Requested --jax-pd, but JAX is unavailable; falling back to NumPy.")

    def denorm(self, i: int, a: float) -> float:
        """Convert normalized action [-1,1] to physical joint range."""
        lo, hi = self.jmin[i], self.jmax[i]
        return float(0.5 * (hi - lo) * (a + 1.0) + lo)

    def step(self, q: np.ndarray, qd: np.ndarray, qdes: np.ndarray) -> np.ndarray:
        """Compute PD control: u = kp*(qdes-q) - kd*qd, clamped to limits.
        
        Args:
            q: Current joint positions (3,)
            qd: Current joint velocities (3,)
            qdes: Desired joint positions (3,)
            
        Returns:
            Control torques (3,) - same format regardless of backend
        """
        if not self._use_jax:
            # NumPy path: Fastest for single-environment use cases
            u = self.kp * (qdes - q) - self.kd * qd
            return _clip(u, self.umin, self.umax)
        
        # JAX path: Optimized for vectorized/batched operations
        # Note: Array conversion overhead makes this slower for single env
        jnp = self._jnp
        u_j = self._pd_step(
            jnp.asarray(q,    dtype=jnp.float32),
            jnp.asarray(qd,   dtype=jnp.float32),
            jnp.asarray(qdes, dtype=jnp.float32),
            self._kp_j, self._kd_j, self._umin_j, self._umax_j
        )
        # Convert back to NumPy for MuJoCo compatibility
        return np.asarray(u_j, dtype=np.float32)


# =====================
# FSM State Machine
# =====================
class FSMStateMachine:
    """
    Finite State Machine for bipedal walking.
    
    Uses foot contact detection and leg pitch angles to determine
    hip and knee target positions for stable walking gait.
    """
    
    # State constants
    HIP_LEG1_SWING = 0  # Left leg forward
    HIP_LEG2_SWING = 1  # Right leg forward
    KNEE_STANCE = 0     # Knee in stance phase
    KNEE_RETRACT = 1    # Knee in retract phase

    def __init__(self):
        self.reset()
        # Cached indices for performance (set by bind_indices)
        self._qpos_hip = None
        self._qpos_lk = None
        self._qpos_rk = None
        self._qvel_hip = None
        self._qvel_lk = None
        self._qvel_rk = None
        self._b_lfoot = None
        self._b_rfoot = None
        self._b_lleg = None
        self._b_rleg = None

    def reset(self):
        """Reset FSM to initial state."""
        self.fsm_hip = self.HIP_LEG2_SWING
        self.fsm_knee1 = self.KNEE_STANCE
        self.fsm_knee2 = self.KNEE_STANCE
        self.hip_qdes = None
        self.lk_qdes = None
        self.rk_qdes = None

    def bind_indices(self, qpos_hip: int, qpos_lk: int, qpos_rk: int,
                     qvel_hip: int, qvel_lk: int, qvel_rk: int,
                     b_lfoot: int, b_rfoot: int, b_lleg: int, b_rleg: int):
        """Bind MuJoCo indices for performance (call once from env.__init__)."""
        self._qpos_hip = qpos_hip
        self._qpos_lk = qpos_lk
        self._qpos_rk = qpos_rk
        self._qvel_hip = qvel_hip
        self._qvel_lk = qvel_lk
        self._qvel_rk = qvel_rk
        self._b_lfoot = b_lfoot
        self._b_rfoot = b_rfoot
        self._b_lleg = b_lleg
        self._b_rleg = b_rleg

    def _leg_pitch(self, data, body_id: int) -> float:
        """Get leg pitch angle from body quaternion (legacy convention)."""
        _, pitch, _ = _quat2euler_zyx_np(data.xquat[body_id])
        return float(-pitch)  # Negative for "leg forward" = positive

    def update(self, data, model, b_lfoot: int = None, b_rfoot: int = None,
               b_lleg: int = None, b_rleg: int = None):
        """
        Update FSM states and compute desired joint positions.
        
        Args:
            data: MuJoCo data object
            model: MuJoCo model object
            b_lfoot, b_rfoot: Left/right foot body IDs (optional if bound)
            b_lleg, b_rleg: Left/right leg body IDs (optional if bound)
        """
        dt = float(model.opt.timestep)

        # Use cached indices if available, otherwise fall back to parameters
        if self._qpos_hip is not None:
            # Fast path: use cached indices
            hip = float(data.qpos[self._qpos_hip])
            lk = float(data.qpos[self._qpos_lk])
            rk = float(data.qpos[self._qpos_rk])
            left_contact = data.xpos[self._b_lfoot, 2] < CONTACT_Z
            right_contact = data.xpos[self._b_rfoot, 2] < CONTACT_Z
            if (self._b_lleg is not None) and (self._b_rleg is not None):
                abs_left = self._leg_pitch(data, self._b_lleg)
                abs_right = self._leg_pitch(data, self._b_rleg)
                
                # Fallback: If leg bodies don't rotate meaningfully, use hip angle
                # Check if left leg pitch is always near zero (identity quaternion issue)
                if abs(abs_left) < 0.01:  # Left leg body not rotating
                    abs_left = -hip  # Use hip angle instead
                if abs(abs_right) < 0.01:  # Right leg body not rotating  
                    abs_right = +hip  # Use hip angle instead
            else:
                abs_left = -hip
                abs_right = +hip
        else:
            # Fallback: use name lookups (slower)
            hip = float(data.qpos[model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip")]])
            lk = float(data.qpos[model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")]])
            rk = float(data.qpos[model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")]])
            left_contact = data.xpos[b_lfoot, 2] < CONTACT_Z
            right_contact = data.xpos[b_rfoot, 2] < CONTACT_Z
            if (b_lleg is not None) and (b_rleg is not None):
                abs_left = self._leg_pitch(data, b_lleg)
                abs_right = self._leg_pitch(data, b_rleg)
                
                # Fallback: If leg bodies don't rotate meaningfully, use hip angle
                if abs(abs_left) < 0.01:  # Left leg body not rotating
                    abs_left = -hip  # Use hip angle instead
                if abs(abs_right) < 0.01:  # Right leg body not rotating  
                    abs_right = +hip  # Use hip angle instead
            else:
                abs_left = -hip
                abs_right = +hip

        # Initialize desired positions on first call
        if self.hip_qdes is None:
            self.hip_qdes = hip
            self.lk_qdes = lk
            self.rk_qdes = rk
            self.fsm_hip = self.HIP_LEG1_SWING if hip < 0.0 else self.HIP_LEG2_SWING

        # Hip state transitions
        if self.fsm_hip == self.HIP_LEG2_SWING and right_contact and (abs_left < 0.0):
            self.fsm_hip = self.HIP_LEG1_SWING
        elif self.fsm_hip == self.HIP_LEG1_SWING and left_contact and (abs_right < 0.0):
            self.fsm_hip = self.HIP_LEG2_SWING

        # Knee state transitions
        # Left knee: retract when right foot lands & left leg forward
        if self.fsm_knee1 == self.KNEE_STANCE and right_contact and (abs_left < 0.0):
            self.fsm_knee1 = self.KNEE_RETRACT
        elif self.fsm_knee1 == self.KNEE_RETRACT and (abs_left > KNEE_RELEASE):
            self.fsm_knee1 = self.KNEE_STANCE

        # Right knee: retract when left foot lands & right leg forward
        if self.fsm_knee2 == self.KNEE_STANCE and left_contact and (abs_right < 0.0):
            self.fsm_knee2 = self.KNEE_RETRACT
        elif self.fsm_knee2 == self.KNEE_RETRACT and (abs_right > KNEE_RELEASE):
            self.fsm_knee2 = self.KNEE_STANCE

        # Compute target positions
        hip_tgt = HIP_SWING_NEG if self.fsm_hip == self.HIP_LEG1_SWING else HIP_SWING_POS
        lk_tgt = KNEE_STANCE if self.fsm_knee1 == self.KNEE_STANCE else KNEE_RETRACT
        rk_tgt = KNEE_STANCE if self.fsm_knee2 == self.KNEE_STANCE else KNEE_RETRACT

        # Apply slew rate limiting (optional)
        self.hip_qdes = _slew(self.hip_qdes, hip_tgt, HIP_SLEW, dt)
        self.lk_qdes = _slew(self.lk_qdes, lk_tgt, KNEE_SLEW, dt)
        self.rk_qdes = _slew(self.rk_qdes, rk_tgt, KNEE_SLEW, dt)

        # Clamp to physical ranges
        self.hip_qdes = float(_clip(self.hip_qdes, JOINT_MIN[0], JOINT_MAX[0]))
        self.lk_qdes = float(_clip(self.lk_qdes, JOINT_MIN[1], JOINT_MAX[1]))
        self.rk_qdes = float(_clip(self.rk_qdes, JOINT_MIN[2], JOINT_MAX[2]))

    def desired_hip(self) -> float:
        """Get desired hip position."""
        return float(self.hip_qdes if self.hip_qdes is not None else 0.0)

    def desired_knees(self) -> tuple[float, float]:
        """Get desired knee positions (left, right)."""
        lk = self.lk_qdes if self.lk_qdes is not None else 0.0
        rk = self.rk_qdes if self.rk_qdes is not None else 0.0
        return float(lk), float(rk)


# Import at end to avoid circular dependencies
import mujoco