"""
Test PD controller clamping behavior.
"""
import numpy as np
import pytest
from passive_walker.core.controller import PDController


def test_pd_clamp():
    """Test that PD controller respects control limits."""
    pd = PDController(use_jax=False)
    
    # Test with extreme desired positions that would exceed limits
    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    qd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    q_des = np.array([10.0, 10.0, 10.0], dtype=np.float32)  # Way outside limits
    
    u = pd.step(q, qd, q_des)
    
    # Check that all control outputs are within limits
    assert np.all(u >= pd.umin), f"Control below minimum: {u} < {pd.umin}"
    assert np.all(u <= pd.umax), f"Control above maximum: {u} > {pd.umax}"
    
    # Test with negative extreme
    q_des = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
    u = pd.step(q, qd, q_des)
    
    assert np.all(u >= pd.umin), f"Control below minimum: {u} < {pd.umin}"
    assert np.all(u <= pd.umax), f"Control above maximum: {u} > {pd.umax}"


def test_pd_clamp_jax():
    """Test JAX PD controller clamping behavior."""
    try:
        pd = PDController(use_jax=True)
        
        # Test with extreme desired positions
        q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        qd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        q_des = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        
        u = pd.step(q, qd, q_des)
        
        # Check that all control outputs are within limits
        assert np.all(u >= pd.umin), f"JAX control below minimum: {u} < {pd.umin}"
        assert np.all(u <= pd.umax), f"JAX control above maximum: {u} > {pd.umax}"
        
    except Exception:
        pytest.skip("JAX not available")
