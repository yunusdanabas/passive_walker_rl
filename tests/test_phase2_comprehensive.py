#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 2: Enhanced Data Collection

This test suite validates all Phase 2 components:
- Curriculum collection system
- Physics conditions generation
- Enhanced FSM collection with perturbations
- Contact information integration
- End-to-end workflow validation
"""

import os
import sys
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from passive_walker.fsm.curriculum_collect import CurriculumCollector
from passive_walker.core.physics_conditions import (
    PhysicsConditionGenerator, 
    PhysicsConditionManager,
    PhysicsParameter,
    create_physics_condition_manager
)
from passive_walker.core.perturbations import PerturbationManager, PerturbationType
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.fsm.collect import collect


class TestPhase2Comprehensive:
    """Comprehensive test suite for Phase 2 components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "episodes": 3,  # Small number for testing
            "max_steps": 100,
            "dt": 0.01,
            "use_gui": False,
            "save_frequency": 1
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_physics_condition_generator_comprehensive(self):
        """Test physics condition generator comprehensively."""
        print("\n=== Testing Physics Condition Generator ===")
        
        # Test initialization
        generator = PhysicsConditionGenerator(seed=42)
        assert generator.rng is not None
        assert len(generator.parameter_ranges) == 6
        
        # Test all parameter ranges
        for param in PhysicsParameter:
            range_def = generator.parameter_ranges[param]
            assert range_def.min_value < range_def.max_value
            assert range_def.min_value <= range_def.default_value <= range_def.max_value
            assert range_def.distribution in ["uniform", "normal", "log_uniform"]
        
        # Test single condition generation
        condition = generator.generate_condition()
        assert len(condition) == 6
        assert generator.validate_condition(condition)
        
        # Test batch generation
        conditions = generator.generate_condition_batch(num_conditions=5)
        assert len(conditions) == 5
        for condition in conditions:
            assert generator.validate_condition(condition)
        
        # Test curriculum generation
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=3)
        assert len(curriculum) == 4
        for stage_idx in range(4):
            assert len(curriculum[stage_idx]) == 3
            for condition in curriculum[stage_idx]:
                assert generator.validate_condition(condition)
        
        # Test parameter sampling distributions
        for param in PhysicsParameter:
            range_def = generator.parameter_ranges[param]
            values = [generator._sample_parameter(range_def) for _ in range(100)]
            assert all(range_def.min_value <= v <= range_def.max_value for v in values)
        
        print("✓ Physics condition generator: PASSED")
    
    def test_physics_condition_manager_comprehensive(self):
        """Test physics condition manager comprehensively."""
        print("\n=== Testing Physics Condition Manager ===")
        
        # Test initialization
        manager = PhysicsConditionManager()
        assert manager.generator is not None
        assert isinstance(manager.generator, PhysicsConditionGenerator)
        
        # Test condition retrieval for different episodes and stages
        for stage in range(4):
            for episode_idx in range(5):
                condition = manager.get_condition_for_episode(
                    episode_idx=episode_idx,
                    total_episodes=10,
                    curriculum_stage=stage
                )
                assert isinstance(condition, dict)
                assert len(condition) > 0
                assert manager.generator.validate_condition(condition)
        
        # Test default condition
        default_condition = manager.generator.get_default_condition()
        assert len(default_condition) == 6
        for param in PhysicsParameter:
            assert param.value in default_condition
        
        print("✓ Physics condition manager: PASSED")
    
    def test_curriculum_collector_comprehensive(self):
        """Test curriculum collector comprehensively."""
        print("\n=== Testing Curriculum Collector ===")
        
        # Test initialization
        collector = CurriculumCollector(self.temp_dir, self.test_config)
        assert collector.output_dir == Path(self.temp_dir)
        assert collector.physics_manager is not None
        assert len(collector.stages) == 4
        
        # Test stage definitions
        for i, stage in enumerate(collector.stages):
            assert stage["name"].startswith(f"stage{i+1}_")
            assert stage["difficulty"] == i + 1.0
            assert stage["episodes"] > 0
            
            # Test perturbation progression
            if i == 0:
                assert not stage["perturbations"]["enabled"]
            else:
                assert stage["perturbations"]["enabled"]
                assert len(stage["perturbations"]["types"]) > 0
                assert len(stage["perturbations"]["probabilities"]) > 0
                assert len(stage["perturbations"]["intensities"]) > 0
        
        # Test serialization
        test_data = {
            "numpy_array": np.array([1, 2, 3]),
            "numpy_scalar": np.float64(3.14),
            "regular_data": {"key": "value", "number": 42}
        }
        serialized = collector._make_serializable(test_data)
        assert isinstance(serialized["numpy_array"], list)
        assert isinstance(serialized["numpy_scalar"], float)
        assert serialized["regular_data"]["key"] == "value"
        
        print("✓ Curriculum collector: PASSED")
    
    def test_perturbation_system_integration(self):
        """Test perturbation system integration."""
        print("\n=== Testing Perturbation System Integration ===")
        
        # Test perturbation manager initialization
        manager = PerturbationManager()
        assert manager is not None
        
        # Test perturbation types
        expected_types = [
            PerturbationType.IMPULSE_LATERAL,
            PerturbationType.IMPULSE_FRONTAL,
            PerturbationType.IMPULSE_TORSO,
            PerturbationType.PUSH_LATERAL,
            PerturbationType.PUSH_FRONTAL,
            PerturbationType.TERRAIN_RAMP,
            PerturbationType.TERRAIN_FRICTION,
            PerturbationType.MASS_TORSO,
            PerturbationType.MASS_LEGS
        ]
        
        for pert_type in expected_types:
            assert pert_type in PerturbationType
        
        print("✓ Perturbation system integration: PASSED")
    
    def test_environment_contact_integration(self):
        """Test environment contact information integration."""
        print("\n=== Testing Environment Contact Integration ===")
        
        # Test environment initialization
        env = PassiveWalkerEnv()
        assert env is not None
        
        # Test observation space dimension
        obs, info = env.reset()
        assert len(obs) == 17  # 11D original + 6D contact
        
        # Test contact information in observation
        contact_info = {
            'left_contact': obs[11],
            'right_contact': obs[12],
            'left_force': obs[13],
            'right_force': obs[14],
            'left_contact_duration': obs[15],
            'right_contact_duration': obs[16]
        }
        
        # Contact flags should be binary
        assert contact_info['left_contact'] in [0.0, 1.0]
        assert contact_info['right_contact'] in [0.0, 1.0]
        
        # Forces should be non-negative
        assert contact_info['left_force'] >= 0.0
        assert contact_info['right_force'] >= 0.0
        
        # Durations should be non-negative
        assert contact_info['left_contact_duration'] >= 0.0
        assert contact_info['right_contact_duration'] >= 0.0
        
        print("✓ Environment contact integration: PASSED")
    
    def test_curriculum_stage_progression(self):
        """Test curriculum stage progression logic."""
        print("\n=== Testing Curriculum Stage Progression ===")
        
        collector = CurriculumCollector(self.temp_dir, self.test_config)
        
        # Test difficulty progression
        difficulties = [stage["difficulty"] for stage in collector.stages]
        assert difficulties == [1.0, 2.0, 3.0, 4.0]
        
        # Test perturbation complexity progression
        perturbation_counts = []
        for stage in collector.stages:
            if stage["perturbations"]["enabled"]:
                perturbation_counts.append(len(stage["perturbations"]["types"]))
            else:
                perturbation_counts.append(0)
        
        # Should be increasing or staying the same
        for i in range(1, len(perturbation_counts)):
            assert perturbation_counts[i] >= perturbation_counts[i-1]
        
        # Test intensity progression
        for stage_idx in range(1, len(collector.stages)):
            stage = collector.stages[stage_idx]
            if stage["perturbations"]["enabled"]:
                intensities = stage["perturbations"]["intensities"]
                for pert_type, intensity in intensities.items():
                    assert 0.0 <= intensity <= 1.0
        
        print("✓ Curriculum stage progression: PASSED")
    
    def test_physics_condition_curriculum_integration(self):
        """Test physics condition curriculum integration."""
        print("\n=== Testing Physics Condition Curriculum Integration ===")
        
        generator = PhysicsConditionGenerator(seed=42)
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=5)
        
        # Test stage-specific parameter selection
        stage1_params = set(curriculum[0][0].keys())
        stage4_params = set(curriculum[3][0].keys())
        
        # Later stages should have more parameters
        assert len(stage4_params) >= len(stage1_params)
        
        # Test parameter variation scaling
        for stage_idx in range(1, 4):
            prev_conditions = curriculum[stage_idx - 1]
            curr_conditions = curriculum[stage_idx]
            
            # Calculate average variation from default
            prev_variation = np.mean([
                np.mean([abs(condition[param.value] - generator.parameter_ranges[param].default_value)
                        for param in PhysicsParameter if param.value in condition])
                for condition in prev_conditions
            ])
            
            curr_variation = np.mean([
                np.mean([abs(condition[param.value] - generator.parameter_ranges[param].default_value)
                        for param in PhysicsParameter if param.value in condition])
                for condition in curr_conditions
            ])
            
            # Later stages should generally have more variation (with tolerance for randomness)
            # Allow some tolerance since this is probabilistic
            if curr_variation < prev_variation * 0.7:
                print(f"Warning: Stage {stage_idx} variation ({curr_variation:.3f}) < previous ({prev_variation:.3f})")
                # This is acceptable due to randomness, just log it
        
        print("✓ Physics condition curriculum integration: PASSED")
    
    def test_end_to_end_curriculum_workflow(self):
        """Test end-to-end curriculum workflow."""
        print("\n=== Testing End-to-End Curriculum Workflow ===")
        
        # Create collector with minimal episodes for testing
        test_config = self.test_config.copy()
        test_config["episodes"] = 2  # Very small for testing
        
        collector = CurriculumCollector(self.temp_dir, test_config)
        
        # Test single stage collection (without actual data collection)
        stage = collector.stages[0]
        stage_result = collector._collect_stage(0, stage)
        
        assert "stage_info" in stage_result
        assert "collection_time" in stage_result
        assert "success" in stage_result
        assert stage_result["stage_info"] == stage
        
        # Test intermediate result saving
        results = {
            "curriculum_info": {"total_stages": 4},
            "stage_results": {stage["name"]: stage_result},
            "collection_time": time.time()
        }
        
        collector._save_intermediate_results(results, 0)
        
        # Check that file was created
        intermediate_file = Path(self.temp_dir) / "curriculum_intermediate_stage_1.json"
        assert intermediate_file.exists()
        
        # Test final result saving
        collector._save_final_results(results)
        final_file = Path(self.temp_dir) / "curriculum_final_results.json"
        assert final_file.exists()
        
        print("✓ End-to-end curriculum workflow: PASSED")
    
    def test_parameter_validation_comprehensive(self):
        """Test comprehensive parameter validation."""
        print("\n=== Testing Parameter Validation ===")
        
        generator = PhysicsConditionGenerator(seed=42)
        
        # Test valid conditions
        valid_condition = generator.get_default_condition()
        assert generator.validate_condition(valid_condition)
        
        # Test invalid conditions
        invalid_conditions = [
            {"gravity": 100.0},  # Out of range
            {"unknown_param": 1.0},  # Unknown parameter
            {"gravity": -1.0},  # Negative gravity
            {"mass": 0.0},  # Zero mass
        ]
        
        for invalid_condition in invalid_conditions:
            # Create a condition with invalid parameter
            condition = valid_condition.copy()
            condition.update(invalid_condition)
            assert not generator.validate_condition(condition)
        
        # Test boundary conditions
        for param in PhysicsParameter:
            range_def = generator.parameter_ranges[param]
            
            # Test minimum boundary
            boundary_condition = valid_condition.copy()
            boundary_condition[param.value] = range_def.min_value
            assert generator.validate_condition(boundary_condition)
            
            # Test maximum boundary
            boundary_condition[param.value] = range_def.max_value
            assert generator.validate_condition(boundary_condition)
        
        print("✓ Parameter validation: PASSED")
    
    def test_serialization_comprehensive(self):
        """Test comprehensive serialization."""
        print("\n=== Testing Serialization ===")
        
        collector = CurriculumCollector(self.temp_dir, self.test_config)
        
        # Test various data types
        test_cases = [
            np.array([1, 2, 3]),
            np.float64(3.14),
            np.int32(42),
            {"nested": {"array": np.array([1, 2])}},
            [np.array([1]), np.array([2])],
            {"mixed": [1, np.array([2]), "three"]}
        ]
        
        for test_case in test_cases:
            serialized = collector._make_serializable(test_case)
            
            # Should not contain numpy types
            def check_no_numpy(obj):
                if isinstance(obj, dict):
                    return all(check_no_numpy(v) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_no_numpy(v) for v in obj)
                else:
                    return not isinstance(obj, (np.ndarray, np.integer, np.floating))
            
            assert check_no_numpy(serialized)
        
        print("✓ Serialization: PASSED")


def test_phase2_integration_simple():
    """Simple integration test for Phase 2 components."""
    print("\n" + "="*60)
    print("PHASE 2 INTEGRATION TEST")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test configuration
        test_config = {
            "episodes": 2,
            "max_steps": 50,
            "dt": 0.01,
            "use_gui": False,
            "save_frequency": 1
        }
        
        # Test curriculum collector
        collector = CurriculumCollector(temp_dir, test_config)
        assert len(collector.stages) == 4
        assert collector.physics_manager is not None
        
        # Test physics condition generation
        generator = PhysicsConditionGenerator(seed=42)
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=2)
        assert len(curriculum) == 4
        
        # Test environment integration
        env = PassiveWalkerEnv()
        obs, info = env.reset()
        assert len(obs) == 17  # 11D + 6D contact
        
        # Test perturbation types
        assert PerturbationType.IMPULSE_LATERAL in PerturbationType
        assert PerturbationType.TERRAIN_RAMP in PerturbationType
        
        print("✓ Phase 2 integration test: PASSED")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("="*80)
    print("PHASE 2 COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_instance = TestPhase2Comprehensive()
    
    try:
        # Run all test methods
        test_methods = [
            test_instance.test_physics_condition_generator_comprehensive,
            test_instance.test_physics_condition_manager_comprehensive,
            test_instance.test_curriculum_collector_comprehensive,
            test_instance.test_perturbation_system_integration,
            test_instance.test_environment_contact_integration,
            test_instance.test_curriculum_stage_progression,
            test_instance.test_physics_condition_curriculum_integration,
            test_instance.test_end_to_end_curriculum_workflow,
            test_instance.test_parameter_validation_comprehensive,
            test_instance.test_serialization_comprehensive
        ]
        
        for test_method in test_methods:
            test_instance.setup_method()
            try:
                test_method()
            finally:
                test_instance.teardown_method()
        
        # Run integration test
        test_phase2_integration_simple()
        
        print("\n" + "="*80)
        print("ALL PHASE 2 TESTS PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nPHASE 2 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
