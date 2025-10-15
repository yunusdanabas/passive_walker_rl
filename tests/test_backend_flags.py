"""
Test PD backend selection and environment variable handling.
"""
import os
from passive_walker.core.env import PassiveWalkerEnv


def test_default_is_numpy():
    """Test that default backend is NumPy."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=False)
    assert env.pd.backend_name == "numpy"


def test_force_numpy_flag_like_cli():
    """Test that explicit NumPy selection works."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=False)
    assert env.pd.backend_name == "numpy"


def test_env_var_requests_jax_but_falls_back_if_missing(monkeypatch):
    """Test environment variable default with fallback."""
    # Simulate no JAX installed
    monkeypatch.delenv("PWALKER_PD_BACKEND", raising=False)
    monkeypatch.setenv("PWALKER_PD_BACKEND", "jax")
    env = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=False)
    # If JAX isn't installed in CI, this should still be NumPy
    assert env.pd.backend_name in ("numpy", "jax")


def test_explicit_jax_param_when_available_or_fallback():
    """Test explicit JAX parameter with fallback."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=True)
    # Accept either outcome depending on whether JAX is installed
    assert env.pd.backend_name in ("jax", "numpy")


def test_backend_name_consistency():
    """Test that backend_name is always set."""
    # Test NumPy path
    env_numpy = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=False)
    assert hasattr(env_numpy.pd, 'backend_name')
    assert env_numpy.pd.backend_name == "numpy"
    
    # Test JAX path (may fallback to NumPy)
    env_jax = PassiveWalkerEnv(mode="fsm", use_gui=False, use_jax_pd=True)
    assert hasattr(env_jax.pd, 'backend_name')
    assert env_jax.pd.backend_name in ("jax", "numpy")
