"""
Reward API:
- get_reward_fn(preset_name, overrides) -> Callable
- Three presets: minimal, default, aggressive
- Returns pure function: reward_fn(signals) -> (reward, extras)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple
import math


@dataclass(frozen=True)
class RewCfg:
    # core
    c_fp: float = 1.0  # forward progress
    c_up: float = 0.5  # upright bonus
    upright_pitch_max: float = 0.25  # radians
    c_ac: float = 3e-4  # action cost (L1)

    # optional shaping
    c_vt: float = 0.25  # velocity tracking
    vx_star: float = 0.8  # m/s
    sigma_v: float = 0.25

    c_sym: float = 0.05  # knee symmetry
    sigma_sym: float = 0.4

    c_fc: float = 0.05  # foot clearance
    foot_clear_target: float = 0.03  # m

    # terminals & clipping
    pen_fall: float = 5.0
    fall_pitch_max: float = 1.0
    fall_z_min: float = 0.15
    clip_low: float = -5.0
    clip_high: float = 5.0


_PRESETS: Dict[str, RewCfg] = {
    "minimal": RewCfg(c_up=0.0, c_ac=3e-4, c_vt=0.0, c_sym=0.0, c_fc=0.0),
    "default": RewCfg(),  # as defined above
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
    if not overrides:
        return cfg
    d = asdict(cfg)
    d.update(overrides)
    return RewCfg(**d)


def get_reward_fn(
    preset: str = "default", overrides: Dict | None = None
) -> Callable[[Dict], Tuple[float, Dict]]:
    base = _PRESETS.get(preset)
    if base is None:
        raise ValueError(f"Unknown reward preset: {preset!r}. Choices: {list(_PRESETS)}")
    cfg = _merge(base, overrides)

    def softplus(x: float) -> float:
        # smooth hinge: ~max(0, x)
        return math.log1p(math.exp(x))

    def reward(signals: Dict) -> Tuple[float, Dict]:
        # required signals (env ensures these exist)
        dx = float(signals["dx"])
        pitch_abs = float(signals["pitch_abs"])
        u_abs_sum = float(signals["u_abs_sum"])
        torso_z = float(signals["torso_z"])
        vx = float(signals.get("vx", 0.0))
        lk_q = float(signals.get("lk_q", 0.0))
        rk_q = float(signals.get("rk_q", 0.0))
        left_z = float(signals.get("left_foot_z", 0.0))
        right_z = float(signals.get("right_foot_z", 0.0))

        # 1) forward progress
        r_fp = cfg.c_fp * dx

        # 2) upright (smooth parabola in [0, upright_pitch_max])
        ratio = pitch_abs / max(1e-6, cfg.upright_pitch_max)
        r_up = cfg.c_up * max(0.0, 1.0 - ratio * ratio)

        # 3) action cost (L1)
        r_ac = cfg.c_ac * u_abs_sum

        # 4) velocity tracking (Gaussian)
        vel_term = 0.0
        if cfg.c_vt != 0.0:
            d = (vx - cfg.vx_star) / max(1e-6, cfg.sigma_v)
            vel_term = cfg.c_vt * math.exp(-0.5 * d * d)

        # 5) symmetry (Gaussian on knee diff)
        sym_term = 0.0
        if cfg.c_sym != 0.0:
            d = (lk_q - rk_q) / max(1e-6, cfg.sigma_sym)
            sym_term = cfg.c_sym * math.exp(-0.5 * d * d)

        # 6) soft foot clearance (softplus around target)
        fc_term = 0.0
        if cfg.c_fc != 0.0:
            lc = softplus(left_z - cfg.foot_clear_target)
            rc = softplus(right_z - cfg.foot_clear_target)
            fc_term = cfg.c_fc * 0.5 * (lc + rc)

        # terminal: fall
        fell = (pitch_abs > cfg.fall_pitch_max) or (torso_z < cfg.fall_z_min)
        pen_fall = cfg.pen_fall if fell else 0.0

        raw = r_fp + r_up + vel_term + sym_term + fc_term - r_ac - pen_fall
        rew = max(cfg.clip_low, min(cfg.clip_high, raw))

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
