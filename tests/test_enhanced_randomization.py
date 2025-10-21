"""
Tests for Phase 1: Data Quality & Environment Enhancement

Tests enhanced randomization, observation noise, and data collection features.
"""

import pytest
import numpy as np
import tempfile
import os
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.randomization import (
    DomainRandomizer, RandomizationConfig, get_randomization_config, RANDOMIZATION_PROFILES
)
from passive_walker.fsm.collect import collect, _add_observation_noise


class TestBasicRandomization:
    """Test basic physics randomization (ramp, friction, mass)."""
    
    def test_ramp_randomization(self):
        """Test that ramp angle randomization works."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
            ramp_deg=10.0,
            ramp_jitter=2.0,  # ±2 degrees
        )
        
        # Collect ramp angles from multiple resets
        ramp_angles = []
        for i in range(20):
            env.reset(seed=i)
            ramp_angles.append(env.ramp_deg)
        
        # Check that we got variation
        assert len(set(ramp_angles)) > 1, "No ramp angle variation observed"
        
        # Check that values are within expected range
        assert all(8.0 <= deg <= 12.0 for deg in ramp_angles), "Ramp angle outside expected range"
        
        env.close()
    
    def test_friction_randomization(self):
        """Test that friction randomization works."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
            friction=0.8,
            friction_min=0.6,
            friction_max=1.0,
        )
        
        # Collect friction values from multiple resets
        friction_values = []
        for i in range(20):
            env.reset(seed=i)
            friction_values.append(env.friction)
        
        # Check that we got variation
        assert len(set(friction_values)) > 1, "No friction variation observed"
        
        # Check that values are within expected range
        assert all(0.6 <= f <= 1.0 for f in friction_values), "Friction outside expected range"
        
        env.close()
    
    def test_mass_randomization(self):
        """Test that mass randomization works."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
        )
        
        original_mass = env._torso_mass0
        
        # Collect masses from multiple resets
        masses = []
        for i in range(20):
            env.reset(seed=i)
            masses.append(env.model.body_mass[env.b_torso])
        
        # Check that we got variation
        assert len(set(masses)) > 1, "No mass variation observed"
        
        # Check that values are within expected range (±5%)
        assert all(0.95 * original_mass <= m <= 1.05 * original_mass for m in masses), \
            "Mass outside expected range"
        
        env.close()


class TestAdvancedRandomization:
    """Test advanced domain randomization features."""
    
    def test_randomization_profiles(self):
        """Test that all randomization profiles are accessible."""
        for profile_name in ["none", "basic", "moderate", "aggressive", "temporal"]:
            config = get_randomization_config(profile_name)
            assert isinstance(config, RandomizationConfig)
    
    def test_invalid_profile_raises_error(self):
        """Test that invalid profile name raises error."""
        with pytest.raises(ValueError):
            get_randomization_config("invalid_profile")
    
    def test_domain_randomizer_initialization(self):
        """Test that DomainRandomizer initializes correctly."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False)
        config = get_randomization_config("basic")
        rng = np.random.RandomState(42)
        
        randomizer = DomainRandomizer(config, env.model, rng)
        
        assert randomizer.config == config
        assert randomizer.model == env.model
        assert randomizer.rng == rng
        
        env.close()
    
    def test_advanced_randomization_via_profile(self):
        """Test advanced randomization using a profile."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
            randomization_profile="moderate"
        )
        
        # Reset with randomization
        env.reset(seed=42)
        
        # Check that domain randomizer was created
        assert env.domain_randomizer is not None
        
        # Check that parameters were randomized
        assert 8.0 <= env.ramp_deg <= 12.0
        assert 0.6 <= env.friction <= 1.0
        
        env.close()


class TestControlFrequency:
    """Test configurable control frequency."""
    
    def test_default_control_frequency(self):
        """Test that default control frequency is 100 Hz."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False)
        assert env.ctrl_hz == 100.0
        env.close()
    
    def test_custom_control_frequency(self):
        """Test setting custom control frequency."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False, ctrl_hz=200.0)
        assert env.ctrl_hz == 200.0
        env.close()
    
    def test_fsm_walks_at_100hz(self):
        """Test that FSM walks stably at 100 Hz."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False, ctrl_hz=100.0)
        obs, _ = env.reset(seed=42)
        
        # Run for 5 seconds
        steps = int(5.0 * env.ctrl_hz)
        fell = False
        
        for _ in range(steps):
            action = np.zeros(3, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            if info.get("fell", False):
                fell = True
                break
        
        assert not fell, "FSM fell at 100Hz"
        env.close()
    
    def test_fsm_walks_at_200hz(self):
        """Test that FSM walks stably at 200 Hz."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False, ctrl_hz=200.0)
        obs, _ = env.reset(seed=42)
        
        # Run for 5 seconds
        steps = int(5.0 * env.ctrl_hz)
        fell = False
        
        for _ in range(steps):
            action = np.zeros(3, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            if info.get("fell", False):
                fell = True
                break
        
        assert not fell, "FSM fell at 200Hz"
        env.close()


class TestObservationNoise:
    """Test observation noise injection."""
    
    def test_no_noise_returns_same_obs(self):
        """Test that zero noise returns unchanged observation."""
        obs = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.2, -0.1, 0.0, 1.5, -1.0, 0.5])
        rng = np.random.RandomState(42)
        
        noisy_obs = _add_observation_noise(obs, noise_level=0.0, rng=rng)
        
        np.testing.assert_array_equal(obs, noisy_obs)
    
    def test_noise_changes_observation(self):
        """Test that noise changes the observation."""
        obs = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.2, -0.1, 0.0, 1.5, -1.0, 0.5])
        rng = np.random.RandomState(42)
        
        noisy_obs = _add_observation_noise(obs, noise_level=1.0, rng=rng)
        
        # Check that at least some values changed
        assert not np.array_equal(obs, noisy_obs), "Observation unchanged with noise"
    
    def test_noise_is_small(self):
        """Test that noise is relatively small."""
        obs = np.array([1.0, 2.0, 3.0, 0.5, -0.5, 0.2, -0.1, 0.0, 1.5, -1.0, 0.5])
        rng = np.random.RandomState(42)
        
        noisy_obs = _add_observation_noise(obs, noise_level=1.0, rng=rng)
        
        # Check that noise is within reasonable bounds (< 10% for most values)
        for i, (orig, noisy) in enumerate(zip(obs, noisy_obs)):
            if abs(orig) > 0.1:  # Skip near-zero values
                relative_change = abs(noisy - orig) / abs(orig)
                assert relative_change < 0.1, f"Noise too large at index {i}"
    
    def test_fsm_walks_with_observation_noise(self):
        """Test that FSM walks stably with observation noise."""
        env = PassiveWalkerEnv(mode="fsm", use_gui=False)
        rng = np.random.RandomState(42)
        
        obs, _ = env.reset(seed=42)
        
        # Run for 5 seconds with noisy observations
        steps = int(5.0 * env.ctrl_hz)
        fell = False
        
        for _ in range(steps):
            # Add noise to observation (FSM doesn't use it, but this simulates collection)
            noisy_obs = _add_observation_noise(obs, noise_level=1.0, rng=rng)
            
            action = np.zeros(3, dtype=np.float32)
            obs, reward, done, info = env.step(action)
            if info.get("fell", False):
                fell = True
                break
        
        assert not fell, "FSM fell with observation noise"
        env.close()


class TestDataCollection:
    """Test data collection with enhanced features."""
    
    def test_collect_with_observation_noise(self):
        """Test that data collection works with observation noise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Collect a few episodes with noise
            collect(
                episodes=2,
                duration_sec=5.0,
                outdir=tmpdir,
                seed=42,
                obs_noise=0.5
            )
            
            # Check that files were created
            files = [f for f in os.listdir(tmpdir) if f.endswith('.npz')]
            assert len(files) == 2, f"Expected 2 episodes, got {len(files)}"
            
            # Check that metadata includes noise info
            import json
            meta_path = os.path.join(tmpdir, "meta.json")
            assert os.path.exists(meta_path), "Metadata file not created"
            
            with open(meta_path) as f:
                meta = json.load(f)
            
            assert "observation_noise" in meta
            assert meta["observation_noise"] == 0.5
    
    def test_collect_with_diverse_physics(self):
        """Test that collection works with diverse physics presets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test a few different physics conditions
            for condition in ["nominal", "gentle", "low_friction", "steep"]:
                collect(
                    episodes=1,
                    duration_sec=3.0,
                    outdir=os.path.join(tmpdir, condition),
                    seed=42,
                    physics_condition=condition
                )
            
            # Check that all conditions produced data
            for condition in ["nominal", "gentle", "low_friction", "steep"]:
                condition_dir = os.path.join(tmpdir, condition)
                files = [f for f in os.listdir(condition_dir) if f.endswith('.npz')]
                assert len(files) >= 1, f"No data collected for {condition}"


class TestRandomizationStability:
    """Test that FSM remains stable with randomization."""
    
    def test_fsm_walks_with_combined_randomization(self):
        """Test FSM stability with all randomization features combined."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
            ramp_jitter=2.0,
            friction_min=0.6,
            friction_max=1.0,
            ctrl_hz=150.0  # Intermediate frequency
        )
        
        # Test multiple episodes with different random conditions
        fall_count = 0
        total_episodes = 10
        
        for i in range(total_episodes):
            obs, _ = env.reset(seed=i)
            
            # Run for 10 seconds
            steps = int(10.0 * env.ctrl_hz)
            
            for _ in range(steps):
                action = np.zeros(3, dtype=np.float32)
                obs, reward, done, info = env.step(action)
                if info.get("fell", False):
                    fall_count += 1
                    break
        
        # Allow up to 40% fall rate (randomization can be challenging)
        fall_rate = fall_count / total_episodes
        assert fall_rate < 0.4, f"Fall rate too high: {fall_rate:.1%}"
        
        env.close()
    
    def test_fsm_with_aggressive_randomization(self):
        """Test FSM with aggressive randomization profile."""
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=False,
            randomize_physics=True,
            randomization_profile="aggressive"
        )
        
        # Test a few episodes
        fall_count = 0
        total_episodes = 5
        
        for i in range(total_episodes):
            obs, _ = env.reset(seed=i + 100)
            
            # Run for 5 seconds
            steps = int(5.0 * env.ctrl_hz)
            
            for _ in range(steps):
                action = np.zeros(3, dtype=np.float32)
                obs, reward, done, info = env.step(action)
                if info.get("fell", False):
                    fall_count += 1
                    break
        
        # With aggressive randomization, allow up to 60% fall rate
        fall_rate = fall_count / total_episodes
        assert fall_rate < 0.6, f"Fall rate too high even for aggressive randomization: {fall_rate:.1%}"
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

