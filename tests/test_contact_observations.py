#!/usr/bin/env python3
"""
Test Suite for Contact Information Enhancement

Tests the enhanced environment with contact information:
- Observation space expansion (11 → 17 dimensions)
- Contact detection and force computation
- Contact duration tracking
- Backward compatibility
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("CONTACT INFORMATION ENHANCEMENT TEST SUITE")
print("=" * 70)

# Test counters
tests_passed = 0
tests_failed = 0
test_errors = []


def test_section(name):
    """Decorator to mark test sections."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)


def run_test(test_name, test_fn):
    """Run a single test and track results."""
    global tests_passed, tests_failed, test_errors
    
    try:
        print(f"\n  ► {test_name}...", end=" ")
        test_fn()
        print("✅ PASS")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"❌ FAIL")
        print(f"    Error: {str(e)}")
        tests_failed += 1
        test_errors.append((test_name, str(e)))
        return False


# =============================================================================
# TEST 1: Environment Initialization
# =============================================================================

test_section("Environment Initialization")

def test_observation_space_dimension():
    """Test that observation space is now 17D."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    # Check observation space dimension
    assert env.observation_space.shape[0] == 17, f"Expected 17D, got {env.observation_space.shape[0]}D"
    
    # Check action space is still 3D
    assert env.action_space.shape[0] == 3, f"Expected 3D action space, got {env.action_space.shape[0]}D"

run_test("Observation space dimension (17D)", test_observation_space_dimension)


def test_contact_variables_initialized():
    """Test that contact tracking variables are initialized."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    # Check contact tracking variables exist
    assert hasattr(env, '_left_contact_duration')
    assert hasattr(env, '_right_contact_duration')
    assert hasattr(env, '_prev_left_contact')
    assert hasattr(env, '_prev_right_contact')
    assert hasattr(env, '_contact_threshold')
    
    # Check initial values
    assert env._left_contact_duration == 0.0
    assert env._right_contact_duration == 0.0
    assert env._prev_left_contact == False
    assert env._prev_right_contact == False
    assert env._contact_threshold == 0.1

run_test("Contact variables initialized", test_contact_variables_initialized)


# =============================================================================
# TEST 2: Observation Generation
# =============================================================================

test_section("Observation Generation")

def test_observation_shape():
    """Test that observations have correct shape."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    assert obs.shape == (17,), f"Expected shape (17,), got {obs.shape}"
    assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

run_test("Observation shape (17,)", test_observation_shape)


def test_observation_components():
    """Test that observation contains all expected components."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Original 11D components (indices 0-10)
    # [x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]
    assert not np.isnan(obs[0])   # x position
    assert not np.isnan(obs[1])   # z position
    assert not np.isnan(obs[2])   # pitch angle
    assert not np.isnan(obs[3])   # x velocity
    assert not np.isnan(obs[4])   # z velocity
    assert not np.isnan(obs[5])   # hip angle
    assert not np.isnan(obs[6])   # left knee slider
    assert not np.isnan(obs[7])   # right knee slider
    assert not np.isnan(obs[8])   # hip angular velocity
    assert not np.isnan(obs[9])   # left knee velocity
    assert not np.isnan(obs[10])  # right knee velocity
    
    # Contact components (indices 11-16)
    # [left_contact, right_contact, left_force, right_force, left_contact_duration, right_contact_duration]
    assert not np.isnan(obs[11])  # left contact flag
    assert not np.isnan(obs[12])  # right contact flag
    assert not np.isnan(obs[13])  # left contact force
    assert not np.isnan(obs[14])  # right contact force
    assert not np.isnan(obs[15])  # left contact duration
    assert not np.isnan(obs[16])  # right contact duration

run_test("Observation components", test_observation_components)


# =============================================================================
# TEST 3: Contact Detection
# =============================================================================

test_section("Contact Detection")

def test_contact_flags_range():
    """Test that contact flags are binary (0 or 1)."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Contact flags should be 0 or 1
    assert obs[11] in [0.0, 1.0], f"Left contact flag should be 0 or 1, got {obs[11]}"
    assert obs[12] in [0.0, 1.0], f"Right contact flag should be 0 or 1, got {obs[12]}"

run_test("Contact flags are binary", test_contact_flags_range)


def test_contact_forces_non_negative():
    """Test that contact forces are non-negative."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Contact forces should be non-negative
    assert obs[13] >= 0.0, f"Left contact force should be >= 0, got {obs[13]}"
    assert obs[14] >= 0.0, f"Right contact force should be >= 0, got {obs[14]}"

run_test("Contact forces non-negative", test_contact_forces_non_negative)


def test_contact_durations_non_negative():
    """Test that contact durations are non-negative."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Contact durations should be non-negative
    assert obs[15] >= 0.0, f"Left contact duration should be >= 0, got {obs[15]}"
    assert obs[16] >= 0.0, f"Right contact duration should be >= 0, got {obs[16]}"

run_test("Contact durations non-negative", test_contact_durations_non_negative)


# =============================================================================
# TEST 4: Contact Method Functionality
# =============================================================================

test_section("Contact Method Functionality")

def test_compute_foot_contacts():
    """Test _compute_foot_contacts method."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env.reset()
    
    contact_info = env._compute_foot_contacts()
    
    # Check all required keys exist
    required_keys = ['left_contact', 'right_contact', 'left_force', 'right_force', 
                     'left_contact_duration', 'right_contact_duration']
    for key in required_keys:
        assert key in contact_info, f"Missing key: {key}"
    
    # Check data types
    assert isinstance(contact_info['left_contact'], float)
    assert isinstance(contact_info['right_contact'], float)
    assert isinstance(contact_info['left_force'], float)
    assert isinstance(contact_info['right_force'], float)
    assert isinstance(contact_info['left_contact_duration'], float)
    assert isinstance(contact_info['right_contact_duration'], float)

run_test("_compute_foot_contacts method", test_compute_foot_contacts)


def test_get_contact_force():
    """Test _get_contact_force method."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env.reset()
    
    # Test with left foot body ID
    left_force = env._get_contact_force(env.b_lfoot)
    assert isinstance(left_force, float)
    assert left_force >= 0.0
    
    # Test with right foot body ID
    right_force = env._get_contact_force(env.b_rfoot)
    assert isinstance(right_force, float)
    assert right_force >= 0.0

run_test("_get_contact_force method", test_get_contact_force)


# =============================================================================
# TEST 5: Contact Duration Tracking
# =============================================================================

test_section("Contact Duration Tracking")

def test_contact_duration_updates():
    """Test that contact durations update over time."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    initial_left_duration = obs[15]
    initial_right_duration = obs[16]
    
    # Step environment
    action = np.array([0.0, 0.0, 0.0])  # No action
    obs, _, _, _, _ = env.step(action)
    
    # Durations should either stay the same or increase
    assert obs[15] >= initial_left_duration, "Left contact duration should not decrease"
    assert obs[16] >= initial_right_duration, "Right contact duration should not decrease"

run_test("Contact duration updates", test_contact_duration_updates)


def test_contact_reset():
    """Test that contact tracking resets properly."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    # Run some steps
    obs, _ = env.reset()
    for _ in range(10):
        action = np.array([0.0, 0.0, 0.0])
        obs, _, _, _, _ = env.step(action)
    
    # Reset environment
    obs, _ = env.reset()
    
    # Contact durations should be reset to 0
    assert obs[15] == 0.0, f"Left contact duration should be 0 after reset, got {obs[15]}"
    assert obs[16] == 0.0, f"Right contact duration should be 0 after reset, got {obs[16]}"

run_test("Contact tracking resets", test_contact_reset)


# =============================================================================
# TEST 6: Integration with Existing Functionality
# =============================================================================

test_section("Integration with Existing Functionality")

def test_fsm_mode_compatibility():
    """Test that FSM mode still works with enhanced observations."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Should be able to step without errors
    action = np.array([0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (17,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

run_test("FSM mode compatibility", test_fsm_mode_compatibility)


def test_research_mode_compatibility():
    """Test that research mode still works with enhanced observations."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="research", use_gui=False)
    obs, _ = env.reset()
    
    # Should be able to step without errors
    action = np.array([0.0, 0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    
    assert obs.shape == (17,)
    assert isinstance(reward, float)

run_test("Research mode compatibility", test_research_mode_compatibility)


# =============================================================================
# TEST 7: Performance and Stability
# =============================================================================

test_section("Performance and Stability")

def test_multiple_steps_stability():
    """Test that multiple steps don't cause errors."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Run many steps
    for i in range(100):
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == (17,)
        assert not np.any(np.isnan(obs)), f"NaN detected in observation at step {i}"
        
        if terminated or truncated:
            break

run_test("Multiple steps stability", test_multiple_steps_stability)


def test_observation_consistency():
    """Test that observations are consistent across steps."""
    from passive_walker.core.env import PassiveWalkerEnv
    
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset()
    
    # Check that observation components are reasonable
    assert -100 < obs[0] < 100, f"X position out of range: {obs[0]}"
    assert -10 < obs[1] < 10, f"Z position out of range: {obs[1]}"
    assert -np.pi < obs[2] < np.pi, f"Pitch angle out of range: {obs[2]}"
    
    # Contact forces should be reasonable
    assert obs[13] < 1000, f"Left contact force too large: {obs[13]}"
    assert obs[14] < 1000, f"Right contact force too large: {obs[14]}"

run_test("Observation consistency", test_observation_consistency)


# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Total Tests: {tests_passed + tests_failed}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")

if tests_failed > 0:
    print("\n" + "="*70)
    print("FAILED TESTS:")
    print("="*70)
    for test_name, error in test_errors:
        print(f"\n  ❌ {test_name}")
        print(f"     {error}")

print("\n" + "="*70)
if tests_failed == 0:
    print("🎉 ALL TESTS PASSED! Contact information enhancement is working!")
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
print("="*70)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)

