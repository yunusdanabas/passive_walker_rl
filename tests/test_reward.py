"""Test reward system functionality."""

from passive_walker.core.reward import get_reward_fn, _PRESETS


def test_reward_presets():
    """Test that all reward presets exist and are valid."""
    assert "minimal" in _PRESETS
    assert "default" in _PRESETS
    assert "aggressive" in _PRESETS

    for preset_name in _PRESETS:
        preset = _PRESETS[preset_name]
        assert preset.c_fp > 0  # Forward progress should always be positive
        assert preset.c_ac >= 0  # Action cost should be non-negative
        assert preset.clip_low < preset.clip_high  # Clipping bounds should be valid


def test_reward_function():
    """Test reward function with fake signals."""
    reward_fn = get_reward_fn("default")

    # Test with minimal signals
    signals = {
        "dx": 0.1,
        "pitch_abs": 0.05,
        "u_abs_sum": 0.5,
        "torso_z": 0.8,
        "vx": 0.2,
        "lk_q": 0.1,
        "rk_q": -0.1,
        "left_foot_z": 0.02,
        "right_foot_z": 0.02,
    }

    reward, extras = reward_fn(signals)

    assert isinstance(reward, (int, float))
    assert isinstance(extras, dict)
    assert "r_forward" in extras
    assert "r_upright" in extras
    assert "r_act_cost" in extras
    assert "fell" in extras


def test_reward_with_overrides():
    """Test reward function with parameter overrides."""
    overrides = {"c_fp": 2.0, "c_up": 1.0}
    reward_fn = get_reward_fn("default", overrides)

    signals = {
        "dx": 0.1,
        "pitch_abs": 0.05,
        "u_abs_sum": 0.5,
        "torso_z": 0.8,
        "vx": 0.2,
        "lk_q": 0.1,
        "rk_q": -0.1,
        "left_foot_z": 0.02,
        "right_foot_z": 0.02,
    }

    reward, extras = reward_fn(signals)
    assert isinstance(reward, (int, float))
    assert isinstance(extras, dict)


def test_reward_fall_handling():
    """Test reward function fall handling."""
    reward_fn = get_reward_fn("default")

    # Test fall condition
    signals = {
        "dx": 0.1,
        "pitch_abs": 1.5,  # Above fall threshold
        "u_abs_sum": 0.5,
        "torso_z": 0.8,
        "vx": 0.2,
        "lk_q": 0.1,
        "rk_q": -0.1,
        "left_foot_z": 0.02,
        "right_foot_z": 0.02,
    }

    reward, extras = reward_fn(signals)
    assert extras["fell"] is True
    assert reward < 0  # Should be penalized for falling
