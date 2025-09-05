"""Test environment functionality."""

import numpy as np
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv


def test_env_creation():
    """Test environment creation and basic properties."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)

    assert env.observation_space.shape == (11,)
    assert env.action_space.shape == (3,)
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0

    env.close()


def test_env_reset():
    """Test environment reset functionality."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)

    obs, info = env.reset()
    assert obs.shape == (11,)
    assert isinstance(info, dict)
    assert not np.any(np.isnan(obs))

    env.close()


def test_env_step():
    """Test environment step functionality."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)

    obs, _ = env.reset()
    action = np.zeros(3, dtype=np.float32)

    for _ in range(10):
        obs, reward, done, info = env.step(action)
        assert obs.shape == (11,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert not np.any(np.isnan(obs))

        if done:
            break

    env.close()


def test_env_modes():
    """Test both FSM and research modes."""
    cfg = load_walker_config("passive_walker/configs/walker.yaml")

    # Test FSM mode
    cfg.mode = "fsm"
    env_fsm = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env_fsm.reset()
    obs, reward, done, info = env_fsm.step(np.zeros(3))
    env_fsm.close()

    # Test research mode
    cfg.mode = "research"
    env_research = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env_research.reset()
    obs, reward, done, info = env_research.step(np.zeros(3))
    env_research.close()

    # Both should work without errors
    assert obs.shape == (11,)
    assert isinstance(reward, (int, float))
