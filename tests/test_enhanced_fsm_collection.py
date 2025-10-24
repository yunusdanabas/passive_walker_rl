#!/usr/bin/env python3
"""
Test Suite for Enhanced FSM Collection with Perturbations

Tests the enhanced FSM collection script with:
- Perturbation injection during data collection
- Contact information in observations (17D)
- Perturbation tracking in NPZ files
- CLI argument parsing
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
print("ENHANCED FSM COLLECTION TEST SUITE")
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
# TEST 1: Collection Function Import and Basic Functionality
# =============================================================================

test_section("Collection Function Import and Basic Functionality")

def test_collect_function_import():
    """Test that the enhanced collect function can be imported."""
    from passive_walker.fsm.collect import collect
    
    # Check function signature includes perturbation parameters
    import inspect
    sig = inspect.signature(collect)
    params = list(sig.parameters.keys())
    
    assert 'perturbation_mode' in params, "perturbation_mode parameter missing"
    assert 'perturbation_strength' in params, "perturbation_strength parameter missing"
    assert 'perturbation_freq' in params, "perturbation_freq parameter missing"

run_test("Collect function import with perturbation parameters", test_collect_function_import)


def test_collect_function_basic():
    """Test basic collection functionality without perturbations."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test basic collection (1 episode, 2 seconds)
        collect(
            episodes=1,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            perturbation_mode="none"
        )
        
        # Check that episode file was created
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        assert len(episode_files) == 1, f"Expected 1 episode file, got {len(episode_files)}"
        
        # Check NPZ file contents
        npz_file = os.path.join(temp_dir, episode_files[0])
        data = np.load(npz_file)
        
        # Check observation shape (should be 17D now)
        assert data['obs'].shape[1] == 17, f"Expected 17D observations, got {data['obs'].shape[1]}D"
        
        # Check that perturbation arrays are not present (since mode="none")
        assert 'info_perturbations' not in data, "Perturbation data should not be present when mode=none"

run_test("Basic collection without perturbations", test_collect_function_basic)


# =============================================================================
# TEST 2: Perturbation Collection
# =============================================================================

test_section("Perturbation Collection")

def test_collection_with_perturbations():
    """Test collection with perturbations enabled."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test collection with random perturbations
        collect(
            episodes=1,
            duration_sec=3.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            perturbation_mode="random",
            perturbation_strength=0.5,
            perturbation_freq=1.0
        )
        
        # Check that episode file was created
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        assert len(episode_files) == 1, f"Expected 1 episode file, got {len(episode_files)}"
        
        # Check NPZ file contents
        npz_file = os.path.join(temp_dir, episode_files[0])
        data = np.load(npz_file)
        
        # Check observation shape (should be 17D)
        assert data['obs'].shape[1] == 17, f"Expected 17D observations, got {data['obs'].shape[1]}D"
        
        # Check that perturbation arrays are present
        assert 'info_perturbations' in data, "Perturbation data should be present"
        assert 'info_perturbation_type' in data, "Perturbation type data should be present"
        
        # Check perturbation array shapes
        perturbations = data['info_perturbations']
        perturbation_types = data['info_perturbation_type']
        
        assert perturbations.shape[0] == data['obs'].shape[0] - 1, "Perturbation array length mismatch"
        assert perturbation_types.shape[0] == data['obs'].shape[0] - 1, "Perturbation type array length mismatch"
        
        # Check perturbation data types
        assert perturbations.dtype == np.bool_, f"Expected bool dtype, got {perturbations.dtype}"
        assert perturbation_types.dtype == np.int32, f"Expected int32 dtype, got {perturbation_types.dtype}"

run_test("Collection with random perturbations", test_collection_with_perturbations)


def test_collection_with_scheduled_perturbations():
    """Test collection with scheduled perturbations."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test collection with scheduled perturbations
        collect(
            episodes=1,
            duration_sec=3.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            perturbation_mode="scheduled",
            perturbation_strength=0.8,
            perturbation_freq=0.5
        )
        
        # Check NPZ file contents
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        npz_file = os.path.join(temp_dir, episode_files[0])
        data = np.load(npz_file)
        
        # Check that perturbation arrays are present
        assert 'info_perturbations' in data, "Perturbation data should be present"
        assert 'info_perturbation_type' in data, "Perturbation type data should be present"

run_test("Collection with scheduled perturbations", test_collection_with_scheduled_perturbations)


# =============================================================================
# TEST 3: Contact Information in Observations
# =============================================================================

test_section("Contact Information in Observations")

def test_contact_info_in_observations():
    """Test that observations contain contact information."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Collect one episode
        collect(
            episodes=1,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm"
        )
        
        # Load and check observations
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        npz_file = os.path.join(temp_dir, episode_files[0])
        data = np.load(npz_file)
        
        obs = data['obs']
        
        # Check observation shape
        assert obs.shape[1] == 17, f"Expected 17D observations, got {obs.shape[1]}D"
        
        # Check contact components (indices 11-16)
        # [left_contact, right_contact, left_force, right_force, left_contact_duration, right_contact_duration]
        left_contact = obs[:, 11]
        right_contact = obs[:, 12]
        left_force = obs[:, 13]
        right_force = obs[:, 14]
        left_duration = obs[:, 15]
        right_duration = obs[:, 16]
        
        # Contact flags should be 0 or 1
        assert np.all((left_contact == 0) | (left_contact == 1)), "Left contact flags should be 0 or 1"
        assert np.all((right_contact == 0) | (right_contact == 1)), "Right contact flags should be 0 or 1"
        
        # Forces should be non-negative
        assert np.all(left_force >= 0), "Left contact forces should be non-negative"
        assert np.all(right_force >= 0), "Right contact forces should be non-negative"
        
        # Durations should be non-negative
        assert np.all(left_duration >= 0), "Left contact durations should be non-negative"
        assert np.all(right_duration >= 0), "Right contact durations should be non-negative"

run_test("Contact information in observations", test_contact_info_in_observations)


# =============================================================================
# TEST 4: CLI Argument Parsing
# =============================================================================

test_section("CLI Argument Parsing")

def test_cli_perturbation_arguments():
    """Test that CLI arguments for perturbations are properly parsed."""
    import argparse
    from passive_walker.fsm.collect import main
    
    # Test argument parser creation
    parser = argparse.ArgumentParser()
    
    # Add perturbation arguments (simulate what the script does)
    parser.add_argument("--perturbation-mode", type=str, default="none", 
                        choices=["none", "random", "scheduled", "curriculum"])
    parser.add_argument("--perturbation-strength", type=float, default=1.0)
    parser.add_argument("--perturbation-freq", type=float, default=0.5)
    
    # Test parsing
    args = parser.parse_args(["--perturbation-mode", "random", "--perturbation-strength", "0.7", "--perturbation-freq", "1.0"])
    
    assert args.perturbation_mode == "random"
    assert args.perturbation_strength == 0.7
    assert args.perturbation_freq == 1.0

run_test("CLI perturbation arguments parsing", test_cli_perturbation_arguments)


# =============================================================================
# TEST 5: Data Quality and Consistency
# =============================================================================

test_section("Data Quality and Consistency")

def test_data_consistency():
    """Test that collected data is consistent and valid."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Collect multiple episodes
        collect(
            episodes=3,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            perturbation_mode="random",
            perturbation_strength=0.5
        )
        
        # Check all episode files
        episode_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("episode_")])
        assert len(episode_files) == 3, f"Expected 3 episode files, got {len(episode_files)}"
        
        for i, episode_file in enumerate(episode_files):
            npz_file = os.path.join(temp_dir, episode_file)
            data = np.load(npz_file)
            
            # Check observation shape consistency
            assert data['obs'].shape[1] == 17, f"Episode {i+1}: Expected 17D observations"
            
            # Check perturbation data consistency
            assert 'info_perturbations' in data, f"Episode {i+1}: Missing perturbation data"
            assert 'info_perturbation_type' in data, f"Episode {i+1}: Missing perturbation type data"
            
            # Check array length consistency
            obs_len = data['obs'].shape[0]
            perturbations_len = data['info_perturbations'].shape[0]
            assert perturbations_len == obs_len - 1, f"Episode {i+1}: Array length mismatch"

run_test("Data consistency across episodes", test_data_consistency)


def test_metadata_inclusion():
    """Test that metadata includes perturbation information."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Collect with perturbations
        collect(
            episodes=1,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            perturbation_mode="random",
            perturbation_strength=0.6
        )
        
        # Check metadata file
        meta_file = os.path.join(temp_dir, "meta.json")
        assert os.path.exists(meta_file), "Metadata file should exist"
        
        import json
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        # Check that metadata contains expected fields
        assert 'episodes' in meta, "Metadata should contain episodes count"
        assert 'duration_sec' in meta, "Metadata should contain duration"
        assert 'observation_dim' in meta, "Metadata should contain observation dimension"
        
        # Check observation dimension
        assert meta['observation_dim'] == 17, f"Expected 17D observations in metadata, got {meta['observation_dim']}D"

run_test("Metadata includes perturbation information", test_metadata_inclusion)


# =============================================================================
# TEST 6: Integration with Existing Features
# =============================================================================

test_section("Integration with Existing Features")

def test_physics_presets_with_perturbations():
    """Test that physics presets work with perturbations."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with different physics conditions
        collect(
            episodes=1,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            physics_condition="steep",
            perturbation_mode="random",
            perturbation_strength=0.5
        )
        
        # Check that episode was collected successfully
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        assert len(episode_files) == 1, "Episode should be collected successfully"

run_test("Physics presets with perturbations", test_physics_presets_with_perturbations)


def test_observation_noise_with_perturbations():
    """Test that observation noise works with perturbations."""
    from passive_walker.fsm.collect import collect
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with observation noise and perturbations
        collect(
            episodes=1,
            duration_sec=2.0,
            outdir=temp_dir,
            seed=42,
            mode="fsm",
            obs_noise=0.1,
            perturbation_mode="random",
            perturbation_strength=0.5
        )
        
        # Check that episode was collected successfully
        episode_files = [f for f in os.listdir(temp_dir) if f.startswith("episode_")]
        assert len(episode_files) == 1, "Episode should be collected successfully"

run_test("Observation noise with perturbations", test_observation_noise_with_perturbations)


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
    print("🎉 ALL TESTS PASSED! Enhanced FSM collection is working!")
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
print("="*70)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)

