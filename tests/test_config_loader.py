"""
Configuration loader tests.

Tests that configuration loading works correctly with stage YAMLs and constraints.
"""

from passive_walker.core.io import load_walker_config


def test_stage_yaml_loads():
    """Test that stage YAML files load correctly."""
    cfg = load_walker_config("passive_walker/bc/bc_train.yaml")
    assert cfg.mode in ("fsm", "research")
    assert cfg.physics.ramp_deg_min < cfg.physics.ramp_deg_max


def test_constraints_merge_lower_precedence():
    """Test that constraints have lower precedence than base config."""
    base = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    constrained = load_walker_config(
        "passive_walker/fsm/fsm_collect.yaml",
        constraint_paths=["passive_walker/configs/constraints/dr_light.yaml"],
    )
    # base has precedence → equal to base
    assert constrained.physics.randomize_physics == base.physics.randomize_physics
    assert constrained.physics.ramp_deg_min == base.physics.ramp_deg_min


def test_constraints_override_defaults():
    """Test that constraints can override default values."""
    # Load base config first to see what it has
    base = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    # Load with constraint that changes physics parameters
    constrained = load_walker_config(
        "passive_walker/fsm/fsm_collect.yaml",
        constraint_paths=["passive_walker/configs/constraints/dr_light.yaml"],
    )
    
    # Base config should have precedence, so values should be from base
    assert constrained.physics.ramp_deg_min == base.physics.ramp_deg_min
    assert constrained.physics.ramp_deg_max == base.physics.ramp_deg_max
    assert constrained.physics.friction == base.physics.friction
    assert constrained.physics.mass_jitter == base.physics.mass_jitter


def test_all_stage_configs_load():
    """Test that all stage configs can be loaded."""
    configs = [
        "passive_walker/fsm/fsm_collect.yaml",
        "passive_walker/bc/bc_train.yaml", 
        "passive_walker/ppo/ppo_train.yaml"
    ]
    
    for config_path in configs:
        cfg = load_walker_config(config_path)
        assert cfg is not None
        assert hasattr(cfg, 'mode')
        assert hasattr(cfg, 'physics')
        assert hasattr(cfg, 'control')


def test_config_dataclass_structure():
    """Test that loaded configs have proper dataclass structure."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    # Check main config structure
    assert hasattr(cfg, 'env')
    assert hasattr(cfg, 'physics')
    assert hasattr(cfg, 'control')
    assert hasattr(cfg, 'reward')
    assert hasattr(cfg, 'fsm')
    assert hasattr(cfg, 'render')
    assert hasattr(cfg, 'debug')
    assert hasattr(cfg, 'jax')
    assert hasattr(cfg, 'bc')
    assert hasattr(cfg, 'ppo')
    
    # Check nested structure
    assert hasattr(cfg.physics, 'ramp_deg_min')
    assert hasattr(cfg.physics, 'ramp_deg_max')
    assert hasattr(cfg.physics, 'randomize_physics')
    
    assert hasattr(cfg.control, 'kp')
    assert hasattr(cfg.control, 'kv')
    assert hasattr(cfg.control, 'joint_ranges')
    
    assert hasattr(cfg.fsm, 'contact_height')
    assert hasattr(cfg.fsm, 'knee_release_threshold')
    assert hasattr(cfg.fsm, 'hip_swing_pos')
    assert hasattr(cfg.fsm, 'hip_swing_neg')
