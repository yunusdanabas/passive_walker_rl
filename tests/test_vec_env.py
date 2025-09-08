"""
Tests for vectorized environment interface.
"""

import numpy as np
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.vec import NumpySubprocVecEnv


def test_vec_reset_step_close():
    """Test basic VecEnv functionality."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    def make():
        return PassiveWalkerEnv(cfg, use_gui=False)
    
    vec = NumpySubprocVecEnv([make, make, make])
    
    # Test reset
    obs, infos = vec.reset(seed=123)
    assert obs.shape[0] == 3 and obs.shape[1] == 11
    assert len(infos) == 3
    
    # Test step
    acts = np.zeros((3, 3), dtype=np.float32)
    o2, r, d, info = vec.step(acts)
    assert o2.shape == obs.shape
    assert r.shape == (3,)
    assert d.shape == (3,)
    assert len(info) == 3
    
    # Test close
    vec.close()


def test_vec_determinism_seed():
    """Test that VecEnv produces deterministic results with same seed."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    def m():
        return PassiveWalkerEnv(cfg, use_gui=False)
    
    v1, v2 = NumpySubprocVecEnv([m, m]), NumpySubprocVecEnv([m, m])
    
    # Test deterministic reset
    o1, _ = v1.reset(seed=42)
    o2, _ = v2.reset(seed=42)
    assert np.allclose(o1, o2, atol=1e-6)
    
    # Test deterministic step
    acts = np.zeros((2, 3), dtype=np.float32)
    o1, r1, d1, _ = v1.step(acts)
    o2, r2, d2, _ = v2.step(acts)
    assert np.allclose(o1, o2, atol=1e-6)
    assert np.allclose(r1, r2, atol=1e-6)
    assert np.array_equal(d1, d2)
    
    v1.close()
    v2.close()


def test_vec_auto_reset():
    """Test that VecEnv auto-resets on done."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    cfg.env.simend = 0.1  # Very short episode
    
    def make():
        return PassiveWalkerEnv(cfg, use_gui=False)
    
    vec = NumpySubprocVecEnv([make])
    
    obs, _ = vec.reset(seed=42)
    acts = np.zeros((1, 3), dtype=np.float32)
    
    # Step until done
    for _ in range(100):  # Should be enough to trigger done
        obs, rew, done, info = vec.step(acts)
        if done[0]:
            # Should have terminal_observation in info
            assert "terminal_observation" in info[0]
            break
    
    vec.close()


def test_vec_single_env():
    """Test VecEnv with single environment."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    
    def make():
        return PassiveWalkerEnv(cfg, use_gui=False)
    
    vec = NumpySubprocVecEnv([make])
    
    obs, infos = vec.reset(seed=123)
    assert obs.shape == (1, 11)
    assert len(infos) == 1
    
    acts = np.zeros((1, 3), dtype=np.float32)
    obs, rew, done, info = vec.step(acts)
    assert obs.shape == (1, 11)
    assert rew.shape == (1,)
    assert done.shape == (1,)
    assert len(info) == 1
    
    vec.close()

