"""
Unit tests for JAX PD controller and reward function parity.

Tests that JAX implementations match NumPy implementations within tolerance.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from passive_walker.jax.controller_jax import pd_step, pd_step_vmap, pd_step_broadcast
from passive_walker.jax.reward_jax import minimal_reward, research_reward
from passive_walker.jax.rng import make_key, split, fold_in, uniform, domain_randomize_physics
from passive_walker.core.reward import get_reward_fn


class TestJAXPDParity:
    """Test JAX PD controller matches NumPy implementation."""
    
    def test_pd_step_scalar_parity(self):
        """Test single PD step matches NumPy implementation."""
        # Random test data
        np.random.seed(42)
        q = np.random.randn(3).astype(np.float32)
        qd = np.random.randn(3).astype(np.float32)
        q_des = np.random.randn(3).astype(np.float32)
        kp = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)
        kv = np.array([1.0, 100.0, 100.0], dtype=np.float32)
        umin = np.array([-50.0, -800.0, -800.0], dtype=np.float32)
        umax = np.array([50.0, 800.0, 800.0], dtype=np.float32)
        
        # NumPy implementation
        u_np = kp * (q_des - q) - kv * qd
        u_np = np.clip(u_np, umin, umax)
        
        # JAX implementation
        u_jax = pd_step(
            jnp.array(q), jnp.array(qd), jnp.array(q_des),
            jnp.array(kp), jnp.array(kv),
            jnp.array(umin), jnp.array(umax)
        )
        
        # Check parity
        np.testing.assert_allclose(u_np, np.array(u_jax), rtol=1e-5, atol=1e-6)
    
    def test_pd_step_batch_parity(self):
        """Test batched PD step matches NumPy implementation."""
        batch_size = 4
        np.random.seed(42)
        
        # Random test data
        q = np.random.randn(batch_size, 3).astype(np.float32)
        qd = np.random.randn(batch_size, 3).astype(np.float32)
        q_des = np.random.randn(batch_size, 3).astype(np.float32)
        kp = np.array([5.0, 1000.0, 1000.0], dtype=np.float32)
        kv = np.array([1.0, 100.0, 100.0], dtype=np.float32)
        umin = np.array([-50.0, -800.0, -800.0], dtype=np.float32)
        umax = np.array([50.0, 800.0, 800.0], dtype=np.float32)
        
        # NumPy implementation (broadcasted)
        u_np = kp[None, :] * (q_des - q) - kv[None, :] * qd
        u_np = np.clip(u_np, umin[None, :], umax[None, :])
        
        # JAX implementation (broadcasted)
        u_jax = pd_step_broadcast(
            jnp.array(q), jnp.array(qd), jnp.array(q_des),
            jnp.array(kp), jnp.array(kv),
            jnp.array(umin), jnp.array(umax)
        )
        
        # Check parity
        np.testing.assert_allclose(u_np, np.array(u_jax), rtol=1e-5, atol=1e-6)
    
    def test_pd_step_vmap_shapes(self):
        """Test vmap PD step returns correct shapes."""
        batch_size = 8
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        q = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        qd = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        q_des = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        kp = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        kv = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        umin = jax.random.normal(subkey, (batch_size, 3))
        key, subkey = jax.random.split(key)
        umax = jax.random.normal(subkey, (batch_size, 3))
        
        u = pd_step_vmap(q, qd, q_des, kp, kv, umin, umax)
        
        assert u.shape == (batch_size, 3)
        assert u.dtype == jnp.float32


class TestJAXRewardParity:
    """Test JAX reward functions match Python implementations."""
    
    def test_minimal_reward_parity(self):
        """Test minimal reward matches Python implementation."""
        # Test data
        dx = 0.1
        cfg = {"c_fp": 1.0, "c_up": 0.5, "upright_pitch_max": 0.25, 
               "c_ac": 3e-4, "pen_fall": 5.0, "fall_pitch_max": 1.0, 
               "fall_z_min": 0.15, "clip_low": -5.0, "clip_high": 5.0}
        
        # Python implementation
        py_reward_fn = get_reward_fn("minimal")
        signals = {
            "dx": dx,
            "pitch_abs": 0.0,
            "u_abs_sum": 0.0,
            "torso_z": 0.8,
            "vx": 0.0,
            "lk_q": 0.0,
            "rk_q": 0.0,
            "left_foot_z": 0.0,
            "right_foot_z": 0.0
        }
        py_reward, py_info = py_reward_fn(signals)
        
        # JAX implementation
        jax_reward, jax_info = minimal_reward(jnp.array(dx), cfg)
        
        # Check parity
        np.testing.assert_allclose(py_reward, float(jax_reward), rtol=1e-5, atol=1e-6)
        assert py_info["fell"] == bool(jax_info["fell"])
    
    def test_research_reward_parity(self):
        """Test research reward matches Python implementation."""
        # Test data
        signals = {
            "dx": 0.1,
            "pitch_abs": 0.2,
            "u_abs_sum": 0.5,
            "torso_z": 0.8,
            "vx": 0.7,
            "lk_q": 0.1,
            "rk_q": -0.1,
            "left_foot_z": 0.05,
            "right_foot_z": 0.03
        }
        cfg = {
            "c_fp": 1.0, "c_up": 0.5, "upright_pitch_max": 0.25,
            "c_ac": 3e-4, "c_vt": 0.25, "vx_star": 0.8, "sigma_v": 0.25,
            "c_sym": 0.05, "sigma_sym": 0.4, "c_fc": 0.05, "foot_clear_target": 0.03,
            "pen_fall": 5.0, "fall_pitch_max": 1.0, "fall_z_min": 0.15,
            "clip_low": -5.0, "clip_high": 5.0
        }
        
        # Python implementation
        py_reward_fn = get_reward_fn("default")
        py_reward, py_info = py_reward_fn(signals)
        
        # JAX implementation
        jax_reward, jax_info = research_reward(
            jnp.array(signals["dx"]),
            jnp.array(signals["pitch_abs"]),
            jnp.array(signals["u_abs_sum"]),
            jnp.array(signals["torso_z"]),
            jnp.array(signals["vx"]),
            jnp.array(signals["lk_q"]),
            jnp.array(signals["rk_q"]),
            jnp.array(signals["left_foot_z"]),
            jnp.array(signals["right_foot_z"]),
            cfg
        )
        
        # Check parity
        np.testing.assert_allclose(py_reward, float(jax_reward), rtol=1e-5, atol=1e-6)
        assert py_info["fell"] == bool(jax_info["fell"])
    
    def test_reward_vmap_shapes(self):
        """Test vmap reward functions return correct shapes."""
        batch_size = 4
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        dx = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        pitch_abs = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        u_abs_sum = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        torso_z = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        vx = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        lk_q = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        rk_q = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        left_foot_z = jax.random.normal(subkey, (batch_size,))
        key, subkey = jax.random.split(key)
        right_foot_z = jax.random.normal(subkey, (batch_size,))
        cfg = {"c_fp": 1.0, "c_up": 0.5, "upright_pitch_max": 0.25, 
               "c_ac": 3e-4, "pen_fall": 5.0, "fall_pitch_max": 1.0, 
               "fall_z_min": 0.15, "clip_low": -5.0, "clip_high": 5.0}
        
        # Test minimal reward vmap
        from passive_walker.jax.reward_jax import minimal_reward_vmap
        reward, info = minimal_reward_vmap(dx, cfg)
        
        assert reward.shape == (batch_size,)
        assert info["fell"].shape == (batch_size,)
        assert reward.dtype == jnp.float32
        assert info["fell"].dtype == jnp.bool_


class TestJAXRNGParity:
    """Test JAX RNG functions work correctly."""
    
    def test_make_key_deterministic(self):
        """Test that make_key produces deterministic results."""
        key1 = make_key(42)
        key2 = make_key(42)
        
        # Same seed should produce same key
        np.testing.assert_array_equal(key1, key2)
    
    def test_split_deterministic(self):
        """Test that split produces deterministic results."""
        key = make_key(42)
        keys1 = split(key, 3)
        keys2 = split(key, 3)
        
        # Same input should produce same output
        for k1, k2 in zip(keys1, keys2):
            np.testing.assert_array_equal(k1, k2)
    
    def test_fold_in_deterministic(self):
        """Test that fold_in produces deterministic results."""
        key = make_key(42)
        new_key1 = fold_in(key, 123)
        new_key2 = fold_in(key, 123)
        
        # Same input should produce same output
        np.testing.assert_array_equal(new_key1, new_key2)
    
    def test_uniform_generation(self):
        """Test uniform random generation."""
        key = make_key(42)
        new_key, values = uniform(key, (5, 3), minval=-1.0, maxval=1.0)
        
        assert values.shape == (5, 3)
        assert jnp.all(values >= -1.0)
        assert jnp.all(values < 1.0)
        assert new_key is not key  # Should return new key
    
    def test_domain_randomize_physics(self):
        """Test domain randomization physics generation."""
        key = make_key(42)
        new_key, params = domain_randomize_physics(
            key, ramp_deg_min=8.0, ramp_deg_max=16.0,
            friction_min=0.7, friction_max=1.1,
            mass_jitter=0.1, base_mass=5.0
        )
        
        assert "gravity" in params
        assert "friction" in params
        assert "mass" in params
        
        # Check ranges
        assert 8.0 <= jnp.rad2deg(jnp.arcsin(params["gravity"][0] / 9.81)) <= 16.0
        assert 0.7 <= params["friction"] <= 1.1
        assert 4.5 <= params["mass"] <= 5.5  # 5.0 ± 10%


if __name__ == "__main__":
    pytest.main([__file__])
