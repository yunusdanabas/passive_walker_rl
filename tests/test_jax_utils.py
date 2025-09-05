"""Test JAX utilities functionality."""

import jax.numpy as jnp
from passive_walker.core.jax_utils import (
    pd_step,
    v_pd_step,
    quat2euler_zyx,
    v_quat2euler_zyx,
    make_batched_reward_fn,
)


def test_pd_step():
    """Test single PD control step."""
    q = jnp.array([0.1, 0.2, 0.3])
    qd = jnp.array([0.01, 0.02, 0.03])
    q_des = jnp.array([0.0, 0.0, 0.0])
    kp = jnp.array([1.0, 2.0, 3.0])
    kv = jnp.array([0.1, 0.2, 0.3])
    umin = jnp.array([-10.0, -20.0, -30.0])
    umax = jnp.array([10.0, 20.0, 30.0])

    u = pd_step(q, qd, q_des, kp, kv, umin, umax)

    assert u.shape == (3,)
    assert jnp.all(u >= umin)
    assert jnp.all(u <= umax)


def test_v_pd_step():
    """Test batched PD control step."""
    q = jnp.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
    qd = jnp.array([[0.01, 0.02, 0.03], [0.02, 0.03, 0.04]])
    q_des = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    kp = jnp.array([1.0, 2.0, 3.0])
    kv = jnp.array([0.1, 0.2, 0.3])
    umin = jnp.array([-10.0, -20.0, -30.0])
    umax = jnp.array([10.0, 20.0, 30.0])

    u = v_pd_step(q, qd, q_des, kp, kv, umin, umax)

    assert u.shape == (2, 3)
    assert jnp.all(u >= umin)
    assert jnp.all(u <= umax)


def test_quat2euler_zyx():
    """Test quaternion to Euler conversion."""
    # Test single quaternion
    q = jnp.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    euler = quat2euler_zyx(q)

    assert euler.shape == (3,)
    assert jnp.allclose(euler, jnp.array([0.0, 0.0, 0.0]), atol=1e-6)

    # Test batch of quaternions
    q_batch = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]])
    euler_batch = quat2euler_zyx(q_batch)

    assert euler_batch.shape == (2, 3)


def test_v_quat2euler_zyx():
    """Test batched quaternion to Euler conversion."""
    q = jnp.array([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]])
    euler = v_quat2euler_zyx(q)

    assert euler.shape == (2, 3)


def test_make_batched_reward_fn():
    """Test batched reward function wrapper."""

    # Create a simple reward function
    def simple_reward(signals):
        dx = signals["dx"]
        pitch_abs = signals["pitch_abs"]
        reward = dx - pitch_abs
        extras = {"r_forward": dx, "r_upright": -pitch_abs}
        return reward, extras

    # Create batched version
    batched_reward = make_batched_reward_fn(simple_reward)

    # Test with batched signals
    signals_batched = {
        "dx": jnp.array([0.1, 0.2, 0.3]),
        "pitch_abs": jnp.array([0.05, 0.1, 0.15]),
    }

    rewards, extras = batched_reward(signals_batched)

    assert rewards.shape == (3,)
    assert isinstance(extras, dict)
    assert "r_forward" in extras
    assert "r_upright" in extras


def test_jax_compilation():
    """Test that JAX functions are properly compiled."""
    # Test that functions are callable and return expected types
    q = jnp.array([0.1, 0.2, 0.3])
    qd = jnp.array([0.01, 0.02, 0.03])
    q_des = jnp.array([0.0, 0.0, 0.0])
    kp = jnp.array([1.0, 2.0, 3.0])
    kv = jnp.array([0.1, 0.2, 0.3])
    umin = jnp.array([-10.0, -20.0, -30.0])
    umax = jnp.array([10.0, 20.0, 30.0])

    # These should not raise errors and should be fast on second call
    u1 = pd_step(q, qd, q_des, kp, kv, umin, umax)
    u2 = pd_step(q, qd, q_des, kp, kv, umin, umax)  # Second call should be fast

    assert jnp.allclose(u1, u2)
