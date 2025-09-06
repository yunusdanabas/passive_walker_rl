from dataclasses import dataclass
from typing import Tuple


@dataclass
class PhysicsCfg:
    ramp_deg_min: float
    ramp_deg_max: float
    friction: Tuple[float, float]
    mass_jitter: float
    fall_z_min: float
    fall_pitch_max: float
    randomize_physics: bool = False


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
class TerminationCfg:
    fall_z_min: float
    fall_pitch_max: float
    max_idle_speed: float = 0.1
    enable_stall_termination: bool = False


@dataclass
class RewardCfg:
    preset: str  # "minimal" | "default" | "aggressive"
    overrides: dict = None  # optional parameter overrides


@dataclass
class FsmCfg:
    contact_height: float = 0.02
    knee_release_threshold: float = 0.01
    hip_swing_pos: float = 0.3
    hip_swing_neg: float = -0.3
    knee_stance: float = 0.0
    knee_retract: float = 0.2


@dataclass
class RenderCfg:
    camera_distance: float = 3.0
    rgb_array_width: int = 640
    rgb_array_height: int = 480


@dataclass
class DebugCfg:
    log_quality: bool = False
    log_fsm: bool = False
    verbose_info: bool = False


@dataclass
class JaxCfg:
    enable: bool = False
    batched: bool = False


@dataclass
class WalkerConfig:
    mode: str  # "fsm" | "research"
    env: EnvCfg
    physics: PhysicsCfg
    control: ControlCfg
    terminations: TerminationCfg
    reward: RewardCfg
    fsm: FsmCfg = FsmCfg()
    render: RenderCfg = RenderCfg()
    debug: DebugCfg = DebugCfg()
    jax: JaxCfg = JaxCfg()
