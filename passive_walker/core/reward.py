# passive_walker/core/reward.py
# Simple reward with minimal knobs. No external config.
# Keeps env compatibility via get_reward_fn(mode).

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

# -------------------- Termination / alive thresholds --------------------
FALL_PITCH_MAX = 1.3    # rad: terminate if |pitch| exceeds this
FALL_Z_MIN     = 0.70   # m: terminate if torso is too low
ALIVE_Z_MIN    = 0.90   # m: small alive bonus if above this

# -------------------- Weights (kept tiny, per mode) --------------------
# Signals used: dx (forward progress), pitch_abs, u_abs_sum, torso_z
WEIGHTS_FSM = dict(
    w_dx=1.0,     # reward for forward progress per step
    w_pitch=0.2,  # penalty for absolute pitch
    w_ctrl=0.001, # penalty for control effort (sum |u|)
    w_alive=0.10, # alive bonus when torso high enough
)

WEIGHTS_RESEARCH = dict(
    w_dx=2.0,
    w_pitch=0.5,
    w_ctrl=0.003,
    w_alive=0.20,
)

def _fell(signals: Dict[str, float]) -> bool:
    """Simple fall detection using pitch and torso height."""
    return (float(signals.get("pitch_abs", 0.0)) > FALL_PITCH_MAX) or (
        float(signals.get("torso_z", 0.0)) < FALL_Z_MIN
    )

def _reward(signals: Dict[str, float], w: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute scalar reward and a tiny info dict."""
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

def compute_reward(signals: Dict[str, float], mode: str = "fsm") -> Tuple[float, Dict[str, float]]:
    """Main entry: pick weights by mode ('fsm' or 'research')."""
    if mode == "research":
        return _reward(signals, WEIGHTS_RESEARCH)
    # default to FSM weights
    return _reward(signals, WEIGHTS_FSM)

def get_reward_fn(mode: str = "fsm"):
    """Env-compatible factory: returns a (signals)->(reward, info) callable."""
    return lambda signals: compute_reward(signals, mode)