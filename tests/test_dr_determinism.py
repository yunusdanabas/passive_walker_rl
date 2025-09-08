"""
Domain randomization determinism tests.

Tests that domain randomization produces deterministic results with the same seed.
"""

import numpy as np
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv


def test_dr_determinism_reset_seed():
    """Test that DR produces identical results with same seed."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    cfg.physics.randomize_physics = True

    env1 = PassiveWalkerEnv(cfg, use_gui=False)
    env2 = PassiveWalkerEnv(cfg, use_gui=False)

    env1.reset(seed=42)
    env2.reset(seed=42)

    g1 = env1.model.opt.gravity.copy()
    g2 = env2.model.opt.gravity.copy()
    f1 = float(env1.model.geom_friction[0,0])
    f2 = float(env2.model.geom_friction[0,0])
    m1 = float(env1.model.body_mass[env1.b_torso])
    m2 = float(env2.model.body_mass[env2.b_torso])

    env1.close(); env2.close()
    assert np.allclose(g1, g2)
    assert f1 == f2
    assert m1 == m2


def test_dr_determinism_different_seeds():
    """Test that DR produces different results with different seeds."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    cfg.physics.randomize_physics = True

    env1 = PassiveWalkerEnv(cfg, use_gui=False)
    env2 = PassiveWalkerEnv(cfg, use_gui=False)

    env1.reset(seed=42)
    env2.reset(seed=123)

    g1 = env1.model.opt.gravity.copy()
    g2 = env2.model.opt.gravity.copy()
    f1 = float(env1.model.geom_friction[0,0])
    f2 = float(env2.model.geom_friction[0,0])
    m1 = float(env1.model.body_mass[env1.b_torso])
    m2 = float(env2.model.body_mass[env2.b_torso])

    env1.close(); env2.close()
    # Should be different (very unlikely to be identical)
    assert not np.allclose(g1, g2, atol=1e-6)
    assert f1 != f2
    assert m1 != m2


def test_dr_determinism_no_randomization():
    """Test that DR is disabled when randomize_physics=False."""
    cfg = load_walker_config("passive_walker/fsm/fsm_collect.yaml")
    cfg.physics.randomize_physics = False

    env1 = PassiveWalkerEnv(cfg, use_gui=False)
    env2 = PassiveWalkerEnv(cfg, use_gui=False)

    env1.reset(seed=42)
    env2.reset(seed=123)  # Different seeds

    g1 = env1.model.opt.gravity.copy()
    g2 = env2.model.opt.gravity.copy()
    f1 = float(env1.model.geom_friction[0,0])
    f2 = float(env2.model.geom_friction[0,0])
    m1 = float(env1.model.body_mass[env1.b_torso])
    m2 = float(env2.model.body_mass[env2.b_torso])

    env1.close(); env2.close()
    # Should be identical (no randomization)
    assert np.allclose(g1, g2)
    assert f1 == f2
    assert m1 == m2

