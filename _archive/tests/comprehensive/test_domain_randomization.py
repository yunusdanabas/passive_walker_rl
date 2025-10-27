"""
Test domain randomization behavior.
"""
import numpy as np
import pytest
from passive_walker.core.env import PassiveWalkerEnv


def test_mass_drift_prevention():
    """Test that mass randomization doesn't drift across resets."""
    # Enable domain randomization
    import passive_walker.core.env as env_module
    original_randomize = env_module.RANDOMIZE_PHYSICS
    env_module.RANDOMIZE_PHYSICS = True
    
    try:
        env = PassiveWalkerEnv(mode="fsm", use_gui=False)
        
        # Get original mass
        original_mass = env._torso_mass0
        
        # Run multiple resets and check mass stays bounded
        masses = []
        for i in range(10):
            env.reset(seed=123 + i)  # Different seeds for different randomizations
            current_mass = float(env.model.body_mass[env.b_torso])
            masses.append(current_mass)
            
            # Mass should be within expected range
            expected_min = original_mass * (1.0 - env_module.MASS_JITTER)
            expected_max = original_mass * (1.0 + env_module.MASS_JITTER)
            assert expected_min <= current_mass <= expected_max, \
                f"Mass {current_mass} outside range [{expected_min}, {expected_max}] at reset {i}"
        
        # Check that masses are properly randomized (not all the same)
        unique_masses = len(set(masses))
        assert unique_masses > 1, "Mass randomization not working - all masses identical"
        
        # Check that mass doesn't drift (each reset should be independent)
        # This is the key test - mass should not compound across resets
        for i, mass in enumerate(masses):
            expected_min = original_mass * (1.0 - env_module.MASS_JITTER)
            expected_max = original_mass * (1.0 + env_module.MASS_JITTER)
            assert expected_min <= mass <= expected_max, \
                f"Mass drift detected: mass {mass} outside bounds at reset {i}"
        
        env.close()
        
    finally:
        # Restore original setting
        env_module.RANDOMIZE_PHYSICS = original_randomize


def test_mass_randomization_disabled():
    """Test that mass stays constant when randomization is disabled."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        original_mass = float(env.model.body_mass[env.b_torso])
        
        # Run multiple resets
        for i in range(5):
            env.reset(seed=123 + i)
            current_mass = float(env.model.body_mass[env.b_torso])
            assert current_mass == original_mass, \
                f"Mass changed from {original_mass} to {current_mass} when randomization disabled"
        
        env.close()
        
    finally:
        env.close()
