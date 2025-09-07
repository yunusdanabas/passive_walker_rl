"""
Reward function module for the passive walker.

Provides configurable reward functions with preset configurations for different training modes.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple
import math


@dataclass(frozen=True)
class RewCfg:
    """Reward configuration parameters."""

    # Core reward components
    c_fp: float = 1.0  # forward progress coefficient
    c_up: float = 0.5  # upright bonus coefficient
    upright_pitch_max: float = 0.25  # max pitch for upright bonus (radians)
    c_ac: float = 3e-4  # action cost coefficient (L1)

    # Optional shaping terms
    c_vt: float = 0.25  # velocity tracking coefficient
    vx_star: float = 0.8  # target velocity (m/s)
    sigma_v: float = 0.25  # velocity tracking width

    c_sym: float = 0.05  # knee symmetry coefficient
    sigma_sym: float = 0.4  # symmetry tracking width

    c_fc: float = 0.05  # foot clearance coefficient
    foot_clear_target: float = 0.03  # target foot clearance (m)

    # Termination and clipping
    pen_fall: float = 5.0  # fall penalty
    fall_pitch_max: float = 1.0  # max pitch before fall (radians)
    fall_z_min: float = 0.15  # min height before fall (m)
    clip_low: float = -5.0  # minimum reward
    clip_high: float = 5.0  # maximum reward


# Reward presets for different training modes
_PRESETS: Dict[str, RewCfg] = {
    "minimal": RewCfg(c_up=0.0, c_ac=3e-4, c_vt=0.0, c_sym=0.0, c_fc=0.0),
    "default": RewCfg(),  # Standard configuration
    "aggressive": RewCfg(
        c_fp=2.0,
        c_up=1.0,
        c_ac=1e-3,
        c_vt=0.5,
        c_sym=0.1,
        c_fc=0.1,
        pen_fall=10.0,
        clip_low=-10.0,
        clip_high=10.0,
    ),
}


def _merge(cfg: RewCfg, overrides: Dict | None) -> RewCfg:
    """Merge configuration with overrides."""
    if not overrides:
        return cfg
    d = asdict(cfg)
    d.update(overrides)
    return RewCfg(**d)


def get_reward_fn(
    preset: str = "default", overrides: Dict | None = None
) -> Callable[[Dict], Tuple[float, Dict]]:
    """Get reward function with specified preset and overrides."""
    base = _PRESETS.get(preset)
    if base is None:
        raise ValueError(f"Unknown reward preset: {preset!r}. Choices: {list(_PRESETS)}")
    cfg = _merge(base, overrides)

    def softplus(x: float) -> float:
        """Smooth hinge function: ~max(0, x)."""
        return math.log1p(math.exp(x))

    def reward(signals: Dict) -> Tuple[float, Dict]:
        """Compute reward from state signals."""
        # Extract required signals
        dx = float(signals["dx"])
        pitch_abs = float(signals["pitch_abs"])
        u_abs_sum = float(signals["u_abs_sum"])
        torso_z = float(signals["torso_z"])
        vx = float(signals.get("vx", 0.0))
        lk_q = float(signals.get("lk_q", 0.0))
        rk_q = float(signals.get("rk_q", 0.0))
        left_z = float(signals.get("left_foot_z", 0.0))
        right_z = float(signals.get("right_foot_z", 0.0))

        # Compute reward components
        r_fp = cfg.c_fp * dx  # Forward progress

        # Upright bonus (smooth parabola)
        ratio = pitch_abs / max(1e-6, cfg.upright_pitch_max)
        r_up = cfg.c_up * max(0.0, 1.0 - ratio * ratio)

        # Action cost (L1 penalty)
        r_ac = cfg.c_ac * u_abs_sum

        # Velocity tracking (Gaussian reward)
        vel_term = 0.0
        if cfg.c_vt != 0.0:
            d = (vx - cfg.vx_star) / max(1e-6, cfg.sigma_v)
            vel_term = cfg.c_vt * math.exp(-0.5 * d * d)

        # Knee symmetry (Gaussian reward)
        sym_term = 0.0
        if cfg.c_sym != 0.0:
            d = (lk_q - rk_q) / max(1e-6, cfg.sigma_sym)
            sym_term = cfg.c_sym * math.exp(-0.5 * d * d)

        # Foot clearance (softplus reward)
        fc_term = 0.0
        if cfg.c_fc != 0.0:
            lc = softplus(left_z - cfg.foot_clear_target)
            rc = softplus(right_z - cfg.foot_clear_target)
            fc_term = cfg.c_fc * 0.5 * (lc + rc)

        # Check for fall and apply penalty
        fell = (pitch_abs > cfg.fall_pitch_max) or (torso_z < cfg.fall_z_min)
        pen_fall = cfg.pen_fall if fell else 0.0

        # Combine all terms and clip
        raw = r_fp + r_up + vel_term + sym_term + fc_term - r_ac - pen_fall
        rew = max(cfg.clip_low, min(cfg.clip_high, raw))

        # Return reward and breakdown
        extras = {
            "r_forward": r_fp,
            "r_upright": r_up,
            "r_vel": vel_term,
            "r_sym": sym_term,
            "r_clear": fc_term,
            "r_act_cost": r_ac,
            "fell": fell,
        }
        return rew, extras

    return reward
