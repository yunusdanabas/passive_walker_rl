import yaml
from .config import WalkerConfig, EnvCfg, PhysicsCfg, ControlCfg, TerminationCfg


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_walker_config(cfg_path: str) -> WalkerConfig:
    raw = load_yaml(cfg_path)
    return WalkerConfig(
        mode=raw["mode"],
        env=EnvCfg(**raw["env"]),
        physics=PhysicsCfg(**raw["physics"]),
        control=ControlCfg(**raw["control"]),
        terminations=TerminationCfg(**raw["terminations"]),
        reward=raw["reward"],
    )


def load_reward_preset(preset_path: str, key: str) -> dict:
    raw = load_yaml(preset_path)
    return raw[key]
