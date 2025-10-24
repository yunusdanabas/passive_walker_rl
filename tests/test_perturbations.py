#!/usr/bin/env python3
"""
Test Suite for Perturbation System

Tests PerturbationManager functionality including:
- Impulse perturbations (instantaneous forces/torques)
- Continuous push perturbations
- Terrain changes (ramp angle, friction)
- Mass distribution changes
- Timing mechanisms and scheduling
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
print("PERTURBATION SYSTEM TEST SUITE")
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
# TEST 1: PerturbationManager Basic Functionality
# =============================================================================

test_section("PerturbationManager Basic Functionality")

def test_perturbation_manager_init():
    """Test PerturbationManager initialization."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationConfig
    
    # Test default initialization
    manager = PerturbationManager()
    assert manager.config is not None
    assert len(manager.active_perturbations) == 0
    assert manager.next_perturbation_time == 0.0
    
    # Test custom config
    config = PerturbationConfig(impulse_force_max=100.0)
    manager = PerturbationManager(config)
    assert manager.config.impulse_force_max == 100.0

run_test("PerturbationManager initialization", test_perturbation_manager_init)


def test_perturbation_config():
    """Test PerturbationConfig dataclass."""
    from passive_walker.core.perturbations import PerturbationConfig
    
    config = PerturbationConfig(
        impulse_force_max=75.0,
        push_force_max=25.0,
        min_interval=2.0,
        max_interval=4.0
    )
    
    assert config.impulse_force_max == 75.0
    assert config.push_force_max == 25.0
    assert config.min_interval == 2.0
    assert config.max_interval == 4.0

run_test("PerturbationConfig", test_perturbation_config)


# =============================================================================
# TEST 2: Impulse Perturbations
# =============================================================================

test_section("Impulse Perturbations")

def test_impulse_lateral():
    """Test lateral impulse perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_impulse(PerturbationType.IMPULSE_LATERAL, strength=0.5)
    
    assert perturbation_id.startswith("impulse_")
    assert len(manager.active_perturbations) == 1
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'impulse'
    assert perturbation_data['perturbation_type'] == PerturbationType.IMPULSE_LATERAL
    assert perturbation_data['strength'] == 0.5
    
    # Check force direction (lateral = x-axis)
    force = perturbation_data['force']
    assert force[0] != 0  # x-component should be non-zero
    assert force[1] == 0  # y-component should be zero
    assert force[2] == 0  # z-component should be zero

run_test("Lateral impulse perturbation", test_impulse_lateral)


def test_impulse_frontal():
    """Test frontal impulse perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_impulse(PerturbationType.IMPULSE_FRONTAL, strength=0.8)
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    
    # Check force direction (frontal = y-axis)
    force = perturbation_data['force']
    assert force[0] == 0  # x-component should be zero
    assert force[1] != 0  # y-component should be non-zero
    assert force[2] == 0  # z-component should be zero

run_test("Frontal impulse perturbation", test_impulse_frontal)


def test_impulse_torso():
    """Test torso torque impulse perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_impulse(PerturbationType.IMPULSE_TORSO, strength=0.6)
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    
    # Check torque direction (torso = z-axis rotation)
    torque = perturbation_data['torque']
    assert torque[0] == 0  # x-component should be zero
    assert torque[1] == 0  # y-component should be zero
    assert torque[2] != 0  # z-component should be non-zero

run_test("Torso torque impulse", test_impulse_torso)


# =============================================================================
# TEST 3: Continuous Push Perturbations
# =============================================================================

test_section("Continuous Push Perturbations")

def test_push_lateral():
    """Test lateral push perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_continuous_push(PerturbationType.PUSH_LATERAL, strength=0.7)
    
    assert perturbation_id.startswith("push_")
    assert len(manager.active_perturbations) == 1
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'push'
    assert perturbation_data['perturbation_type'] == PerturbationType.PUSH_LATERAL
    assert perturbation_data['duration'] > 0  # Should have positive duration
    
    # Check force direction (lateral = x-axis)
    force = perturbation_data['force']
    assert force[0] != 0  # x-component should be non-zero
    assert force[1] == 0  # y-component should be zero
    assert force[2] == 0  # z-component should be zero

run_test("Lateral push perturbation", test_push_lateral)


def test_push_frontal():
    """Test frontal push perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_continuous_push(PerturbationType.PUSH_FRONTAL, strength=0.9)
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    
    # Check force direction (frontal = y-axis)
    force = perturbation_data['force']
    assert force[0] == 0  # x-component should be zero
    assert force[1] != 0  # y-component should be non-zero
    assert force[2] == 0  # z-component should be zero

run_test("Frontal push perturbation", test_push_frontal)


def test_push_duration():
    """Test push perturbation duration."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    
    # Test custom duration
    custom_duration = 3.0
    perturbation_id = manager.add_continuous_push(
        PerturbationType.PUSH_LATERAL, 
        strength=0.5, 
        duration=custom_duration
    )
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['duration'] == custom_duration

run_test("Push perturbation duration", test_push_duration)


# =============================================================================
# TEST 4: Terrain Perturbations
# =============================================================================

test_section("Terrain Perturbations")

def test_terrain_ramp():
    """Test terrain ramp angle perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_terrain_change(PerturbationType.TERRAIN_RAMP, strength=0.8)
    
    assert perturbation_id.startswith("terrain_")
    assert len(manager.active_perturbations) == 1
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'terrain'
    assert perturbation_data['perturbation_type'] == PerturbationType.TERRAIN_RAMP
    assert 'terrain_params' in perturbation_data
    assert 'ramp_angle' in perturbation_data['terrain_params']

run_test("Terrain ramp perturbation", test_terrain_ramp)


def test_terrain_friction():
    """Test terrain friction perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_terrain_change(PerturbationType.TERRAIN_FRICTION, strength=0.6)
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'terrain'
    assert perturbation_data['perturbation_type'] == PerturbationType.TERRAIN_FRICTION
    assert 'terrain_params' in perturbation_data
    assert 'friction' in perturbation_data['terrain_params']

run_test("Terrain friction perturbation", test_terrain_friction)


# =============================================================================
# TEST 5: Mass Perturbations
# =============================================================================

test_section("Mass Perturbations")

def test_mass_torso():
    """Test torso mass change perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_mass_change(PerturbationType.MASS_TORSO, strength=0.5)
    
    assert perturbation_id.startswith("mass_")
    assert len(manager.active_perturbations) == 1
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'mass'
    assert perturbation_data['perturbation_type'] == PerturbationType.MASS_TORSO
    assert 'mass_params' in perturbation_data
    assert 'torso_mass_change' in perturbation_data['mass_params']

run_test("Torso mass perturbation", test_mass_torso)


def test_mass_legs():
    """Test leg mass change perturbation."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    perturbation_id = manager.add_mass_change(PerturbationType.MASS_LEGS, strength=0.7)
    
    perturbation_data = manager.active_perturbations[perturbation_id]
    assert perturbation_data['type'] == 'mass'
    assert perturbation_data['perturbation_type'] == PerturbationType.MASS_LEGS
    assert 'mass_params' in perturbation_data
    assert 'leg_mass_change' in perturbation_data['mass_params']

run_test("Leg mass perturbation", test_mass_legs)


# =============================================================================
# TEST 6: Timing and Scheduling
# =============================================================================

test_section("Timing and Scheduling")

def test_perturbation_scheduling():
    """Test perturbation timing and scheduling."""
    from passive_walker.core.perturbations import PerturbationManager
    
    manager = PerturbationManager()
    
    # Test initial state
    assert manager.next_perturbation_time == 0.0
    assert manager.last_perturbation_time == 0.0
    
    # Test scheduling
    manager._schedule_next_perturbation(5.0)
    assert manager.next_perturbation_time > 5.0
    assert manager.last_perturbation_time == 5.0

run_test("Perturbation scheduling", test_perturbation_scheduling)


def test_reset():
    """Test perturbation manager reset."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    
    # Add some perturbations
    manager.add_impulse(PerturbationType.IMPULSE_LATERAL)
    manager.add_continuous_push(PerturbationType.PUSH_FRONTAL)
    manager._schedule_next_perturbation(10.0)
    
    # Verify perturbations exist
    assert len(manager.active_perturbations) == 2
    assert manager.next_perturbation_time > 0
    
    # Reset
    manager.reset()
    
    # Verify reset
    assert len(manager.active_perturbations) == 0
    assert manager.next_perturbation_time == 0.0
    assert manager.last_perturbation_time == 0.0

run_test("Perturbation manager reset", test_reset)


# =============================================================================
# TEST 7: Statistics and Info
# =============================================================================

test_section("Statistics and Info")

def test_get_stats():
    """Test perturbation statistics."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    
    # Test empty stats
    stats = manager.get_stats()
    assert stats['active_perturbations'] == 0
    assert stats['next_perturbation_time'] == 0.0
    assert stats['last_perturbation_time'] == 0.0
    assert len(stats['perturbation_types']) == 0
    
    # Add perturbations and test again
    manager.add_impulse(PerturbationType.IMPULSE_LATERAL)
    manager.add_continuous_push(PerturbationType.PUSH_FRONTAL)
    manager._schedule_next_perturbation(5.0)
    
    stats = manager.get_stats()
    assert stats['active_perturbations'] == 2
    assert stats['next_perturbation_time'] > 0
    assert stats['last_perturbation_time'] == 5.0
    assert len(stats['perturbation_types']) == 2

run_test("Perturbation statistics", test_get_stats)


# =============================================================================
# TEST 8: Factory Function
# =============================================================================

test_section("Factory Function")

def test_create_perturbation_manager():
    """Test perturbation manager factory function."""
    from passive_walker.core.perturbations import create_perturbation_manager
    
    # Test different modes
    manager_none = create_perturbation_manager("none")
    manager_random = create_perturbation_manager("random")
    manager_scheduled = create_perturbation_manager("scheduled")
    manager_curriculum = create_perturbation_manager("curriculum", strength=0.8)
    
    assert manager_none is not None
    assert manager_random is not None
    assert manager_scheduled is not None
    assert manager_curriculum is not None
    
    # Test scheduled mode has fixed intervals
    assert manager_scheduled.config.min_interval == 2.0
    assert manager_scheduled.config.max_interval == 2.0

run_test("Perturbation manager factory", test_create_perturbation_manager)


# =============================================================================
# TEST 9: Mock Environment Integration
# =============================================================================

test_section("Mock Environment Integration")

class MockEnvironment:
    """Mock environment for testing perturbation application."""
    
    def __init__(self):
        self.data = MockData()
        self.model = MockModel()
        self._perturbation_ramp_angle = 0.0
        self._perturbation_friction = 1.0
        self._perturbation_torso_mass_change = 0.0
        self._perturbation_leg_mass_change = 0.0


class MockData:
    """Mock MuJoCo data."""
    
    def __init__(self):
        self.time = 0.0
        self.xfrc_applied = np.zeros((10, 6))  # 10 bodies, 6 DOF


class MockModel:
    """Mock MuJoCo model."""
    
    def __init__(self):
        self.body_names = ['torso', 'left_foot', 'right_foot']
        self._bodies = {'torso': MockBody(0), 'left_foot': MockBody(1), 'right_foot': MockBody(2)}
    
    def body(self, name):
        return self._bodies[name]


class MockBody:
    """Mock MuJoCo body."""
    
    def __init__(self, body_id):
        self.id = body_id


def test_mock_environment_integration():
    """Test perturbation application with mock environment."""
    from passive_walker.core.perturbations import PerturbationManager, PerturbationType
    
    manager = PerturbationManager()
    env = MockEnvironment()
    
    # Test impulse application
    manager.add_impulse(PerturbationType.IMPULSE_LATERAL, strength=0.5)
    perturbation_data = list(manager.active_perturbations.values())[0]
    
    # Apply perturbation
    result = manager._apply_impulse(env, perturbation_data)
    assert result == False  # Impulse should be removed after application
    
    # Test push application
    manager.add_continuous_push(PerturbationType.PUSH_FRONTAL, strength=0.7, duration=2.0)
    perturbation_data = list(manager.active_perturbations.values())[0]
    perturbation_data['start_time'] = 0.0
    
    # Apply push (should continue)
    env.data.time = 1.0
    result = manager._apply_push(env, perturbation_data)
    assert result == True  # Push should continue
    
    # Apply push (should end)
    env.data.time = 3.0
    result = manager._apply_push(env, perturbation_data)
    assert result == False  # Push should end

run_test("Mock environment integration", test_mock_environment_integration)


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
    print("🎉 ALL TESTS PASSED! Perturbation system is working!")
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
print("="*70)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)

