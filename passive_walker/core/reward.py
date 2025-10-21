# passive_walker/core/reward.py
# Enhanced reward system with research mode for RL training.
# Keeps env compatibility via get_reward_fn(mode).

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# -------------------- Termination / alive thresholds --------------------
FALL_PITCH_MAX = 1.3    # rad: terminate if |pitch| exceeds this
FALL_Z_MIN     = 0.70   # m: terminate if torso is too low
ALIVE_Z_MIN    = 0.90   # m: small alive bonus if above this

# -------------------- FSM mode weights (simple) --------------------
FSM_WEIGHTS = dict(
    w_dx=2.0,     # reward for forward progress per step
    w_pitch=0.1,  # penalty for absolute pitch
    w_ctrl=0.0005, # penalty for control effort (sum |u|)
    w_alive=0.20, # alive bonus when torso high enough
)

# -------------------- Research mode weights (enhanced) --------------------
RESEARCH_WEIGHTS = dict(
    # Core terms
    w_dx=1.0,           # forward progress
    w_pitch=0.1,        # posture penalty
    w_ctrl=0.0015,      # control effort
    w_alive=0.5,        # upright bonus
    
    # Enhanced terms
    w_velocity=0.3,     # velocity tracking bonus
    w_symmetry=0.2,    # left-right symmetry bonus
    w_foot_clear=0.2,   # foot clearance bonus
    w_smooth=0.1,       # smooth motion penalty
)

# -------------------- Research mode parameters --------------------
RESEARCH_PARAMS = dict(
    target_velocity=0.15,    # target forward velocity (m/s)
    velocity_sigma=0.05,     # velocity tracking width
    symmetry_sigma=0.1,      # symmetry penalty width
    foot_clear_target=0.03,  # target foot clearance (m)
    upright_pitch_max=0.2,   # max pitch for upright bonus (rad)
)

def _fell(signals: Dict[str, float]) -> bool:
    """Simple fall detection using pitch and torso height."""
    return (float(signals.get("pitch_abs", 0.0)) > FALL_PITCH_MAX) or (
        float(signals.get("torso_z", 0.0)) < FALL_Z_MIN
    )

def _fsm_reward(signals: Dict[str, float], w: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute simple FSM reward."""
    dx       = float(signals.get("dx", 0.0))
    pitch    = float(signals.get("pitch_abs", 0.0))
    u_sum    = float(signals.get("u_abs_sum", 0.0))
    torso_z  = float(signals.get("torso_z", 0.0))

    alive = w["w_alive"] if torso_z >= ALIVE_Z_MIN else 0.0
    r = (
        w["w_dx"]   * max(dx, 0.0)      # forward only
        - w["w_pitch"] * pitch          # posture penalty
        - w["w_ctrl"]  * u_sum          # effort penalty
        + alive                          # small shaping
    )

    info = {
        "fell": _fell(signals),
        "r_dx": w["w_dx"] * max(dx, 0.0),
        "r_pitch": -w["w_pitch"] * pitch,
        "r_ctrl": -w["w_ctrl"] * u_sum,
        "r_alive": alive,
    }

    return float(r), info


def _research_reward(signals: Dict[str, float], w: Dict[str, float], p: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute enhanced research reward with additional shaping."""
    # Core signals
    dx = float(signals.get("dx", 0.0))
    pitch_abs = float(signals.get("pitch_abs", 0.0))
    u_sum = float(signals.get("u_abs_sum", 0.0))
    torso_z = float(signals.get("torso_z", 0.0))
    
    # Enhanced signals
    velocity_x = float(signals.get("velocity_x", 0.0))
    left_knee_pos = float(signals.get("left_knee_pos", 0.0))
    right_knee_pos = float(signals.get("right_knee_pos", 0.0))
    left_foot_z = float(signals.get("left_foot_z", 0.0))
    right_foot_z = float(signals.get("right_foot_z", 0.0))
    
    # 1. Forward progress
    forward_reward = w["w_dx"] * max(dx, 0.0)
    
    # 2. Smooth upright bonus
    upright_ratio = pitch_abs / p["upright_pitch_max"]
    upright_bonus = w["w_alive"] * max(0.0, 1.0 - upright_ratio**2)
    
    # 3. Control effort penalty
    control_penalty = w["w_ctrl"] * u_sum
    
    # 4. Velocity tracking bonus
    vel_diff = (velocity_x - p["target_velocity"]) / p["velocity_sigma"]
    velocity_bonus = w["w_velocity"] * np.exp(-vel_diff**2 / 2.0)
    
    # 5. Symmetry bonus (penalize left-right imbalance)
    knee_diff = (left_knee_pos - right_knee_pos) / p["symmetry_sigma"]
    symmetry_bonus = w["w_symmetry"] * np.exp(-knee_diff**2 / 2.0)
    
    # 6. Foot clearance bonus
    def soft_hinge(x):
        return np.log1p(np.exp(x))  # smooth approximation of max(0, x)
    
    left_clear = soft_hinge(left_foot_z - p["foot_clear_target"])
    right_clear = soft_hinge(right_foot_z - p["foot_clear_target"])
    foot_clearance_bonus = w["w_foot_clear"] * 0.5 * (left_clear + right_clear)
    
    # 7. Smooth motion penalty (penalize jerky control)
    control_change = float(signals.get("u_change_sum", 0.0))
    smooth_penalty = w["w_smooth"] * control_change
    
    # Total reward
    r = (forward_reward + upright_bonus + velocity_bonus + 
         symmetry_bonus + foot_clearance_bonus - 
         control_penalty - smooth_penalty)
    
    info = {
        "fell": _fell(signals),
        "r_dx": forward_reward,
        "r_upright": upright_bonus,
        "r_velocity": velocity_bonus,
        "r_symmetry": symmetry_bonus,
        "r_foot_clear": foot_clearance_bonus,
        "r_ctrl": -control_penalty,
        "r_smooth": -smooth_penalty,
    }
    
    return float(r), info

def compute_reward(signals: Dict[str, float], mode: str = "fsm") -> Tuple[float, Dict[str, float]]:
    """Main entry: route to appropriate reward function based on mode."""
    if mode == "research":
        return _research_reward(signals, RESEARCH_WEIGHTS, RESEARCH_PARAMS)
    else:
        return _fsm_reward(signals, FSM_WEIGHTS)

def get_reward_fn(mode: str = "fsm"):
    """Env-compatible factory: returns a (signals)->(reward, info) callable."""
    return lambda signals: compute_reward(signals, mode)