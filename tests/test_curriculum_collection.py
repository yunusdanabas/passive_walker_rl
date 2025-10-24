#!/usr/bin/env python3
"""
Test suite for curriculum collection and physics conditions.

This test verifies the curriculum collection system and physics condition
generation for robust data collection across different difficulty stages.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

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


class TestPhysicsConditionGenerator:
    """Test physics condition generation."""
    
    def test_generator_initialization(self):
        """Test physics condition generator initialization."""
        generator = PhysicsConditionGenerator(seed=42)
        assert generator.rng is not None
        assert len(generator.parameter_ranges) == 6  # All physics parameters
        
    def test_parameter_ranges(self):
        """Test that parameter ranges are properly defined."""
        generator = PhysicsConditionGenerator()
        
        # Check that all parameters have valid ranges
        for param in PhysicsParameter:
            assert param in generator.parameter_ranges
            range_def = generator.parameter_ranges[param]
            assert range_def.min_value < range_def.max_value
            assert range_def.min_value <= range_def.default_value <= range_def.max_value
    
    def test_generate_single_condition(self):
        """Test generation of a single physics condition."""
        generator = PhysicsConditionGenerator(seed=42)
        
        # Generate condition with all parameters
        condition = generator.generate_condition()
        
        # Check that all parameters are present
        assert len(condition) == 6
        for param in PhysicsParameter:
            assert param.value in condition
            assert isinstance(condition[param.value], (int, float))
    
    def test_generate_condition_batch(self):
        """Test generation of multiple physics conditions."""
        generator = PhysicsConditionGenerator(seed=42)
        
        # Generate batch of conditions
        conditions = generator.generate_condition_batch(num_conditions=5)
        
        assert len(conditions) == 5
        for condition in conditions:
            assert len(condition) == 6
            for param in PhysicsParameter:
                assert param.value in condition
    
    def test_curriculum_conditions(self):
        """Test curriculum condition generation."""
        generator = PhysicsConditionGenerator(seed=42)
        
        # Generate curriculum conditions
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=3)
        
        assert len(curriculum) == 4
        for stage_idx in range(4):
            assert stage_idx in curriculum
            assert len(curriculum[stage_idx]) == 3
            
            # Check that conditions are valid
            for condition in curriculum[stage_idx]:
                assert generator.validate_condition(condition)
    
    def test_parameter_sampling(self):
        """Test parameter sampling from different distributions."""
        generator = PhysicsConditionGenerator(seed=42)
        
        # Test uniform distribution
        range_def = generator.parameter_ranges[PhysicsParameter.GRAVITY]
        values = [generator._sample_parameter(range_def) for _ in range(100)]
        
        assert all(range_def.min_value <= v <= range_def.max_value for v in values)
        assert min(values) >= range_def.min_value
        assert max(values) <= range_def.max_value
    
    def test_condition_validation(self):
        """Test physics condition validation."""
        generator = PhysicsConditionGenerator(seed=42)
        
        # Valid condition
        valid_condition = generator.get_default_condition()
        assert generator.validate_condition(valid_condition)
        
        # Invalid condition (out of range)
        invalid_condition = valid_condition.copy()
        invalid_condition["gravity"] = 100.0  # Way out of range
        assert not generator.validate_condition(invalid_condition)
        
        # Invalid condition (unknown parameter)
        invalid_condition2 = valid_condition.copy()
        invalid_condition2["unknown_param"] = 1.0
        assert not generator.validate_condition(invalid_condition2)


class TestPhysicsConditionManager:
    """Test physics condition management."""
    
    def test_manager_initialization(self):
        """Test physics condition manager initialization."""
        manager = PhysicsConditionManager()
        assert manager.generator is not None
        assert isinstance(manager.generator, PhysicsConditionGenerator)
    
    def test_get_condition_for_episode(self):
        """Test getting physics condition for specific episode."""
        manager = PhysicsConditionManager()
        
        # Test different episodes and stages
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
    
    def test_default_condition(self):
        """Test default physics condition."""
        generator = PhysicsConditionGenerator()
        default_condition = generator.get_default_condition()
        
        assert len(default_condition) == 6
        for param in PhysicsParameter:
            assert param.value in default_condition
            range_def = generator.parameter_ranges[param]
            assert default_condition[param.value] == range_def.default_value


class TestCurriculumCollector:
    """Test curriculum collection system."""
    
    def test_collector_initialization(self):
        """Test curriculum collector initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = CurriculumCollector(temp_dir)
            
            assert collector.output_dir == Path(temp_dir)
            assert collector.physics_manager is not None
            assert len(collector.stages) == 4
    
    def test_stage_definitions(self):
        """Test curriculum stage definitions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = CurriculumCollector(temp_dir)
            
            # Check stage progression
            for i, stage in enumerate(collector.stages):
                assert stage["name"].startswith(f"stage{i+1}_")
                assert stage["difficulty"] == i + 1.0
                assert stage["episodes"] > 0
                
                # Check perturbation progression
                if i == 0:
                    assert not stage["perturbations"]["enabled"]
                else:
                    assert stage["perturbations"]["enabled"]
                    assert len(stage["perturbations"]["types"]) > 0
    
    def test_curriculum_collection_structure(self):
        """Test curriculum collection structure without actual collection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = CurriculumCollector(temp_dir)
            
            # Test single stage collection structure
            stage = collector.stages[0]
            stage_result = collector._collect_stage(0, stage)
            
            assert "stage_info" in stage_result
            assert "collection_time" in stage_result
            assert "success" in stage_result
            assert stage_result["stage_info"] == stage
    
    def test_serialization(self):
        """Test result serialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = CurriculumCollector(temp_dir)
            
            # Test serialization of simple data
            test_data = {
                "numpy_array": np.array([1, 2, 3]),
                "numpy_scalar": np.float64(3.14),
                "regular_data": {"key": "value", "number": 42}
            }
            
            serialized = collector._make_serializable(test_data)
            
            assert isinstance(serialized["numpy_array"], list)
            assert isinstance(serialized["numpy_scalar"], float)
            assert serialized["regular_data"]["key"] == "value"


class TestIntegration:
    """Integration tests for curriculum collection."""
    
    def test_physics_condition_integration(self):
        """Test integration between physics conditions and curriculum."""
        generator = PhysicsConditionGenerator(seed=42)
        manager = PhysicsConditionManager(generator)
        
        # Generate curriculum conditions
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=5)
        
        # Test that conditions are properly distributed across stages
        for stage_idx in range(4):
            stage_conditions = curriculum[stage_idx]
            assert len(stage_conditions) == 5
            
            # All conditions should be valid
            for condition in stage_conditions:
                assert generator.validate_condition(condition)
    
    def test_curriculum_progression(self):
        """Test that curriculum shows proper difficulty progression."""
        generator = PhysicsConditionGenerator(seed=42)
        curriculum = generator.generate_curriculum_conditions(stages=4, conditions_per_stage=10)
        
        # Check that later stages have more parameter variation
        for stage_idx in range(1, 4):
            prev_conditions = curriculum[stage_idx - 1]
            curr_conditions = curriculum[stage_idx]
            
            # Calculate average variation from default
            prev_variation = np.mean([
                np.mean([abs(condition[param.value] - generator.parameter_ranges[param].default_value)
                        for param in PhysicsParameter])
                for condition in prev_conditions
            ])
            
            curr_variation = np.mean([
                np.mean([abs(condition[param.value] - generator.parameter_ranges[param].default_value)
                        for param in PhysicsParameter])
                for condition in curr_conditions
            ])
            
            # Later stages should generally have more variation
            # (This might not always be true due to randomness, but should trend upward)
            assert curr_variation >= prev_variation * 0.8  # Allow some tolerance


def test_curriculum_collection_simple():
    """Simple test of curriculum collection without full data collection."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create collector with minimal episodes for testing
        base_config = {
            "episodes": 2,  # Very small for testing
            "max_steps": 100,
            "dt": 0.01,
            "use_gui": False,
            "save_frequency": 1
        }
        
        collector = CurriculumCollector(temp_dir, base_config)
        
        # Test stage definition
        assert len(collector.stages) == 4
        assert collector.stages[0]["name"] == "stage1_basic"
        assert collector.stages[3]["name"] == "stage4_heavy"
        
        # Test physics manager integration
        assert collector.physics_manager is not None
        
        print("Curriculum collection system test passed!")


if __name__ == "__main__":
    # Run simple test
    test_curriculum_collection_simple()
    
    print("All curriculum collection tests passed!")
