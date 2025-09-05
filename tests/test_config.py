"""Test configuration loading and dataclass parsing."""

from passive_walker.core.io import load_walker_config, load_reward_preset
from passive_walker.core.config import WalkerConfig


def test_config_loading():
    """Test YAML config loading into dataclasses."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")

    assert isinstance(cfg, WalkerConfig)
    assert cfg.mode in ["fsm", "research"]
    assert cfg.env.simend > 0
    assert cfg.env.ctrl_hz > 0
    assert len(cfg.control.kp) == 3
    assert len(cfg.control.kv) == 3
    assert cfg.physics.ramp_deg_min > 0
    assert cfg.physics.ramp_deg_max > cfg.physics.ramp_deg_min


def test_reward_preset_loading():
    """Test reward preset loading."""
    preset = load_reward_preset("passive_walker/configs/reward_presets.yaml", "default")

    assert isinstance(preset, dict)
    assert "c_fp" in preset
    assert "c_up" in preset
    assert "c_ac" in preset
    assert preset["c_fp"] > 0


def test_config_overrides():
    """Test that config overrides work correctly."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")

    # Test that we can modify config values
    original_mode = cfg.mode
    cfg.mode = "research"
    assert cfg.mode == "research"
    cfg.mode = original_mode
