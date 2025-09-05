from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class PhysicsCfg:
    ramp_deg_min: float
    ramp_deg_max: float
    friction: Tuple[float, float]
    mass_jitter: float
    fall_z_min: float
    fall_pitch_max: float


@dataclass
class ControlCfg:
    kp: Tuple[float, float, float]
    kv: Tuple[float, float, float]
    umin: Tuple[float, float, float]
    umax: Tuple[float, float, float]
    joint_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    use_nn_for_hip: bool
    use_nn_for_knees: bool


@dataclass
class EnvCfg:
    simend: float
    ctrl_hz: int
    xml_path: str
    randomize_physics: bool


@dataclass
class RewardCfg:
    # keep keys flexible – we'll map presets later
    params: dict


@dataclass
class WalkerConfig:
    mode: str               # "fsm" | "research"
    env: EnvCfg
    physics: PhysicsCfg
    control: ControlCfg
    reward: str             # preset key
