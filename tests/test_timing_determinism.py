"""
Test timing fidelity and determinism.
"""
import numpy as np
import pytest
from passive_walker.core.env import PassiveWalkerEnv


def test_timing_fidelity():
    """Test that control timing stays within timestep tolerance."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env._timing_debug = True  # Enable timing debug
    
    try:
        obs, _ = env.reset(seed=123)
        zero_action = np.zeros(3, dtype=np.float32)
        
        # Run for 100 steps and check timing
        for step in range(100):
            obs, reward, done, info = env.step(zero_action)
            
            # Check that timing error stays reasonable (within control period + small tolerance)
            if hasattr(env, '_t_next'):
                time_error = abs(env.data.time - env._t_next)
                control_period = 1.0 / env.ctrl_hz
                tolerance = 1e-6  # Small tolerance for floating point precision
                assert time_error <= control_period + tolerance, f"Timing error {time_error} > control_period+tolerance {control_period+tolerance} at step {step}"
            
            if done:
                break
                
    finally:
        env.close()


def test_determinism():
    """Test that identical seeds produce identical trajectories."""
    # Run two episodes with same seed
    env1 = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env2 = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        # Reset both with same seed
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        # Check initial observations are identical
        np.testing.assert_array_equal(obs1, obs2, "Initial observations differ")
        
        # Run first 50 steps and compare
        zero_action = np.zeros(3, dtype=np.float32)
        for step in range(50):
            obs1, reward1, done1, info1 = env1.step(zero_action)
            obs2, reward2, done2, info2 = env2.step(zero_action)
            
            # Check observations are identical
            np.testing.assert_array_equal(obs1, obs2, f"Observations differ at step {step}")
            assert reward1 == reward2, f"Rewards differ at step {step}: {reward1} vs {reward2}"
            assert done1 == done2, f"Done flags differ at step {step}: {done1} vs {done2}"
            
            # Check key info values are identical
            for key in ["dx", "pitch_abs", "torso_z"]:
                assert info1[key] == info2[key], f"Info[{key}] differs at step {step}"
            
            if done1 or done2:
                break
                
    finally:
        env1.close()
        env2.close()


def test_deterministic_collector():
    """Test that collector produces identical results with same seed."""
    import tempfile
    from pathlib import Path
    from passive_walker.fsm.collect import collect
    import hashlib
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Collect two episodes with same seed
        outdir1 = Path(tmpdir) / "collect1"
        outdir2 = Path(tmpdir) / "collect2"
        outdir1.mkdir()
        outdir2.mkdir()
        
        collect(episodes=1, steps=64, outdir=str(outdir1), seed=123)
        collect(episodes=1, steps=64, outdir=str(outdir2), seed=123)
        
        # Load both NPZ files
        data1 = np.load(outdir1 / "episode_000000.npz")
        data2 = np.load(outdir2 / "episode_000000.npz")
        
        # Check all arrays are identical
        for key in data1.files:
            assert key in data2.files, f"Key {key} missing in second collection"
            np.testing.assert_array_equal(data1[key], data2[key], 
                                        f"Array {key} differs between collections")
        
        # Also check file hashes are identical
        hash1 = hashlib.md5(open(outdir1 / "episode_000000.npz", "rb").read()).hexdigest()
        hash2 = hashlib.md5(open(outdir2 / "episode_000000.npz", "rb").read()).hexdigest()
        assert hash1 == hash2, "NPZ file hashes differ despite same seed"
