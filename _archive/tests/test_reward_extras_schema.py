"""
Test reward function extras schema.
"""
import numpy as np
import pytest
from passive_walker.core.reward import compute_reward


def test_reward_extras_schema():
    """Test that reward function returns correct extras keys."""
    signals = {
        "dx": 0.1,
        "pitch_abs": 0.05,
        "u_abs_sum": 10.0,
        "torso_z": 1.2,
    }
    
    reward, info = compute_reward(signals, mode="fsm")
    
    # Check required keys exist
    required_keys = ["r_dx", "r_pitch", "r_ctrl", "r_alive", "fell"]
    for key in required_keys:
        assert key in info, f"Missing reward key: {key}"
    
    # Check values are finite
    for key in required_keys:
        value = info[key]
        assert np.isfinite(value), f"Non-finite value for {key}: {value}"
    
    # Check reward is finite
    assert np.isfinite(reward), f"Non-finite reward: {reward}"
    
    # Check fell is boolean
    assert isinstance(info["fell"], bool), f"fell should be boolean: {type(info['fell'])}"


def test_reward_extras_research_mode():
    """Test reward extras in research mode."""
    signals = {
        "dx": 0.2,
        "pitch_abs": 0.1,
        "u_abs_sum": 20.0,
        "torso_z": 1.1,
    }
    
    reward, info = compute_reward(signals, mode="research")
    
    # Check required keys exist
    required_keys = ["r_dx", "r_pitch", "r_ctrl", "r_alive", "fell"]
    for key in required_keys:
        assert key in info, f"Missing reward key: {key}"
    
    # Check values are finite
    for key in required_keys:
        value = info[key]
        assert np.isfinite(value), f"Non-finite value for {key}: {value}"
