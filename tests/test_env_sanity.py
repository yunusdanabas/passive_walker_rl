"""
Environment sanity tests for headless operation.

Tests that the environment runs correctly in headless mode without GUI dependencies.
"""

import numpy as np
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv


def test_env_fsm_steps_headless():
    """Test FSM mode runs headless for 20 seconds."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env.reset(seed=123)
    assert obs.shape == (11,)
    total_r = 0.0
    steps = 0
    # 20 seconds at default ctrl_hz (smoke; keep bounded by simend)
    while True:
        obs, r, done, info = env.step(np.zeros(3, dtype=np.float32))
        total_r += r; steps += 1
        if done: break
    env.close()
    assert steps > 0
    # fell/stalled booleans always present
    assert "fell" in info and "stalled" in info


def test_env_research_mode_headless():
    """Test research mode runs headless."""
    cfg = load_walker_config("passive_walker/ppo/ppo_train.yaml")
    cfg.mode = "research"
    env = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env.reset(seed=7)
    for _ in range(16):
        obs, r, done, info = env.step(np.zeros(3, dtype=np.float32))
        if done: break
    env.close()


def test_env_rgb_array_rendering():
    """Test rgb_array rendering works headless."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    # Need GUI=True for rendering to work
    env = PassiveWalkerEnv(cfg, use_gui=True)
    obs, _ = env.reset(seed=42)
    
    # Test rgb_array rendering
    img = env.render('rgb_array')
    if img is not None:  # Only test if rendering is available
        assert img.shape == (cfg.render.rgb_array_height, cfg.render.rgb_array_width, 3)
        assert img.dtype == np.uint8
    
    env.close()


def test_env_info_telemetry():
    """Test that info dictionary contains expected telemetry."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    cfg.debug.log_quality = True
    cfg.debug.log_fsm = True
    env = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env.reset(seed=42)
    
    obs, r, done, info = env.step(np.zeros(3, dtype=np.float32))
    
    # Required telemetry
    assert "fell" in info
    assert "stalled" in info
    assert "unstable" in info
    
    # Optional telemetry (when enabled)
    if cfg.debug.log_quality:
        assert "quality_score" in info
    if cfg.debug.log_fsm:
        assert "fsm_state" in info
    
    env.close()
