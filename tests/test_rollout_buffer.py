"""Test rollout buffer functionality."""

import pytest
import numpy as np
import tempfile
import os
from passive_walker.core.rollout_buffer import RolloutBuffer, MultiEnvRolloutBuffer


def test_rollout_buffer_creation():
    """Test rollout buffer creation."""
    buffer = RolloutBuffer(rollout_len=100, obs_dim=11, act_dim=3, store_extras=True)

    assert buffer.rollout_len == 100
    assert buffer.obs_dim == 11
    assert buffer.act_dim == 3
    assert buffer.store_extras is True
    assert buffer.size() == 0
    assert not buffer.is_full()


def test_rollout_buffer_add():
    """Test adding data to rollout buffer."""
    buffer = RolloutBuffer(rollout_len=10, obs_dim=11, act_dim=3, store_extras=True)

    # Add some test data
    for i in range(5):
        obs = np.random.randn(11).astype(np.float32)
        act = np.random.randn(3).astype(np.float32)
        rew = float(i * 0.1)
        done = i == 4
        info = {"test": i, "fell": False}
        extras = {"r_forward": rew, "r_upright": 0.5, "fell": False}

        buffer.add(obs, act, rew, done, info, extras)

    assert buffer.size() == 5
    assert not buffer.is_full()


def test_rollout_buffer_overflow():
    """Test rollout buffer overflow handling."""
    buffer = RolloutBuffer(rollout_len=3, obs_dim=11, act_dim=3, store_extras=True)

    # Fill buffer
    for i in range(3):
        obs = np.random.randn(11).astype(np.float32)
        act = np.random.randn(3).astype(np.float32)
        buffer.add(obs, act, 0.0, False, {}, {})

    assert buffer.is_full()

    # Adding more should raise error
    with pytest.raises(RuntimeError):
        buffer.add(obs, act, 0.0, False, {}, {})


def test_rollout_buffer_get():
    """Test getting data from rollout buffer."""
    buffer = RolloutBuffer(rollout_len=5, obs_dim=11, act_dim=3, store_extras=True)

    # Add test data
    test_obs = []
    test_acts = []
    for i in range(3):
        obs = np.random.randn(11).astype(np.float32)
        act = np.random.randn(3).astype(np.float32)
        test_obs.append(obs)
        test_acts.append(act)
        buffer.add(obs, act, 0.0, False, {}, {})

    data = buffer.get()

    assert "obs" in data
    assert "act" in data
    assert "rew" in data
    assert "done" in data
    assert "info" in data
    assert "extras" in data

    assert data["obs"].shape == (3, 11)
    assert data["act"].shape == (3, 3)
    assert data["rew"].shape == (3,)
    assert data["done"].shape == (3,)
    assert data["info"].shape == (3,)


def test_rollout_buffer_normalization():
    """Test observation normalization."""
    buffer = RolloutBuffer(rollout_len=10, obs_dim=11, act_dim=3, store_extras=True)

    # Add test data
    for i in range(5):
        obs = np.random.randn(11).astype(np.float32)
        act = np.random.randn(3).astype(np.float32)
        buffer.add(obs, act, 0.0, False, {}, {})

    # Test normalization
    norm_obs = buffer.get_normalized_obs()
    assert norm_obs.shape == (5, 11)
    assert not np.any(np.isnan(norm_obs))

    # Test norm stats
    mean, std, count = buffer.get_norm_stats()
    assert mean.shape == (11,)
    assert std.shape == (11,)
    assert count == 5


def test_rollout_buffer_save_load():
    """Test saving and loading rollout buffer."""
    buffer = RolloutBuffer(rollout_len=5, obs_dim=11, act_dim=3, store_extras=True)

    # Add test data
    for i in range(3):
        obs = np.random.randn(11).astype(np.float32)
        act = np.random.randn(3).astype(np.float32)
        buffer.add(obs, act, 0.0, False, {}, {})

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = f.name

    try:
        buffer.save_npz(temp_path)

        # Load back
        loaded_data = RolloutBuffer.load_npz(temp_path)

        # Verify data integrity
        original_data = buffer.get()
        assert np.allclose(original_data["obs"], loaded_data["obs"])
        assert np.allclose(original_data["act"], loaded_data["act"])
        assert np.allclose(original_data["rew"], loaded_data["rew"])

    finally:
        os.unlink(temp_path)


def test_multi_env_rollout_buffer():
    """Test multi-environment rollout buffer."""
    multi_buf = MultiEnvRolloutBuffer(num_envs=3, rollout_len=10, obs_dim=11, act_dim=3)

    # Add data to different environments
    for env_idx in range(3):
        for step in range(5):
            obs = np.random.randn(11).astype(np.float32)
            act = np.random.randn(3).astype(np.float32)
            multi_buf.add(env_idx, obs, act, 0.0, False, {})

    # Test individual buffer access
    for env_idx in range(3):
        data = multi_buf.get(env_idx)
        assert data["obs"].shape == (5, 11)
        assert data["act"].shape == (5, 3)

    # Test stacked data
    stacked = multi_buf.stacked()
    assert stacked["obs"].shape == (3, 5, 11)
    assert stacked["act"].shape == (3, 5, 3)
