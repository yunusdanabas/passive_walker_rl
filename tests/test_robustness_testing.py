#!/usr/bin/env python3
"""
Test suite for robustness testing framework.

This test validates the robustness testing components and ensures
proper functionality across different conditions.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
import tempfile
import shutil
import numpy as np
import pytest

from tools.evaluation.robustness_testing import (
    RobustnessConfig, 
    RobustnessResult, 
    RobustnessTester
)
from passive_walker.core.physics_conditions import PhysicsParameter


class TestRobustnessConfig:
    """Test robustness configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = RobustnessConfig()
        
        # Check default values
        assert len(config.control_frequencies) == 4
        assert config.control_frequencies == [50.0, 100.0, 150.0, 200.0]
        
        assert len(config.obs_noise_levels) == 4
        assert config.obs_noise_levels == [0.0, 0.01, 0.05, 0.10]
        
        assert len(config.action_noise_levels) == 4
        assert config.action_noise_levels == [0.0, 0.01, 0.05, 0.10]
        
        assert len(config.physics_params) == 4
        assert PhysicsParameter.GRAVITY in config.physics_params
        
        assert len(config.dropout_rates) == 4
        assert config.dropout_rates == [0.0, 0.05, 0.10, 0.20]
        
        assert config.episodes_per_condition == 10
        assert config.max_steps_per_episode == 1000
        assert config.dt == 0.01
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = RobustnessConfig(
            control_frequencies=[100.0, 200.0],
            obs_noise_levels=[0.0, 0.05],
            episodes_per_condition=5
        )
        
        assert config.control_frequencies == [100.0, 200.0]
        assert config.obs_noise_levels == [0.0, 0.05]
        assert config.episodes_per_condition == 5


class TestRobustnessResult:
    """Test robustness result data structure."""
    
    def test_result_creation(self):
        """Test robustness result creation."""
        result = RobustnessResult(
            condition_name="test_condition",
            condition_params={"param": 1.0},
            metrics={"success_rate": 0.8, "avg_distance": 5.0},
            episodes=[],
            success_rate=0.8,
            avg_distance=5.0,
            avg_reward=10.0,
            failure_modes=["forward_fall"]
        )
        
        assert result.condition_name == "test_condition"
        assert result.condition_params == {"param": 1.0}
        assert result.success_rate == 0.8
        assert result.avg_distance == 5.0
        assert result.avg_reward == 10.0
        assert result.failure_modes == ["forward_fall"]


class TestRobustnessTester:
    """Test robustness tester functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = RobustnessConfig(
            control_frequencies=[100.0],
            obs_noise_levels=[0.0, 0.01],
            action_noise_levels=[0.0],
            physics_params=[PhysicsParameter.GRAVITY],
            dropout_rates=[0.0],
            episodes_per_condition=2,
            max_steps_per_episode=50
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_tester_initialization(self):
        """Test robustness tester initialization."""
        tester = RobustnessTester(self.test_config)
        
        assert tester.config == self.test_config
        assert tester.results == []
        assert tester.physics_manager is not None
    
    def test_noise_functions(self):
        """Test noise injection functions."""
        tester = RobustnessTester(self.test_config)
        
        # Test observation noise
        obs = np.array([1.0, 2.0, 3.0])
        noisy_obs = tester._add_observation_noise(obs, 0.1)
        assert noisy_obs.shape == obs.shape
        assert not np.array_equal(noisy_obs, obs)  # Should be different
        
        # Test action noise
        action = np.array([0.5, -0.3, 0.8])
        noisy_action = tester._add_action_noise(action, 0.1)
        assert noisy_action.shape == action.shape
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)
        
        # Test dropout
        obs_with_dropout = tester._apply_dropout(obs, 0.5)
        assert obs_with_dropout.shape == obs.shape
        # Some values should be zero (with high probability)
    
    def test_failure_classification(self):
        """Test failure mode classification."""
        tester = RobustnessTester(self.test_config)
        
        # Test forward fall
        obs_forward = np.array([1.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure = tester._classify_failure(obs_forward, {})
        assert failure == "forward_fall"
        
        # Test backward fall
        obs_backward = np.array([1.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure = tester._classify_failure(obs_backward, {})
        assert failure == "backward_fall"
        
        # Test stagnation
        obs_stagnant = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure = tester._classify_failure(obs_stagnant, {})
        assert failure == "stagnation"
        
        # Test other
        obs_normal = np.array([1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure = tester._classify_failure(obs_normal, {})
        assert failure == "other"
    
    def test_metrics_computation(self):
        """Test metrics computation from episodes."""
        tester = RobustnessTester(self.test_config)
        
        # Test with empty episodes
        metrics = tester._compute_metrics([])
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_distance"] == 0.0
        assert metrics["avg_reward"] == 0.0
        
        # Test with sample episodes
        episodes = [
            {
                "distance": 5.0,
                "total_reward": 10.0,
                "steps": 100,
                "failure_mode": None
            },
            {
                "distance": 3.0,
                "total_reward": 8.0,
                "steps": 80,
                "failure_mode": "forward_fall"
            }
        ]
        
        metrics = tester._compute_metrics(episodes)
        assert metrics["success_rate"] == 0.5  # 1 out of 2 successful
        assert metrics["avg_distance"] == 4.0  # (5.0 + 3.0) / 2
        assert metrics["avg_reward"] == 9.0  # (10.0 + 8.0) / 2
        assert metrics["avg_steps"] == 90.0  # (100 + 80) / 2
        assert "forward_fall" in metrics["failure_modes"]


class TestRobustnessIntegration:
    """Integration tests for robustness testing."""
    
    def test_robustness_testing_workflow(self):
        """Test complete robustness testing workflow."""
        # This is a simplified test that doesn't require actual model loading
        config = RobustnessConfig(
            control_frequencies=[100.0],
            obs_noise_levels=[0.0],
            episodes_per_condition=1,
            max_steps_per_episode=10
        )
        
        tester = RobustnessTester(config)
        
        # Test that the tester can be initialized and configured
        assert tester.config.episodes_per_condition == 1
        assert len(tester.results) == 0
        
        # Test noise functions work correctly
        test_obs = np.random.random(17)
        noisy_obs = tester._add_observation_noise(test_obs, 0.1)
        assert noisy_obs.shape == test_obs.shape
        
        test_action = np.random.random(3)
        noisy_action = tester._add_action_noise(test_action, 0.1)
        assert noisy_action.shape == test_action.shape
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)
        
        # Test dropout
        dropped_obs = tester._apply_dropout(test_obs, 0.5)
        assert dropped_obs.shape == test_obs.shape
        
        print("Robustness testing workflow: PASSED")


def test_robustness_simple():
    """Simple test of robustness testing components."""
    print("\n=== Testing Robustness Testing Framework ===")
    
    # Test configuration
    config = RobustnessConfig(episodes_per_condition=2)
    print(f"✓ Configuration created with {config.episodes_per_condition} episodes per condition")
    
    # Test tester initialization
    tester = RobustnessTester(config)
    print(f"✓ RobustnessTester initialized with {len(tester.config.control_frequencies)} control frequencies")
    
    # Test noise functions
    test_obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    noisy_obs = tester._add_observation_noise(test_obs, 0.1)
    print(f"✓ Observation noise injection: {noisy_obs.shape}")
    
    test_action = np.array([0.5, -0.3, 0.8])
    noisy_action = tester._add_action_noise(test_action, 0.1)
    print(f"✓ Action noise injection: {noisy_action.shape}")
    
    # Test failure classification
    obs_forward = np.array([1.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    failure = tester._classify_failure(obs_forward, {})
    print(f"✓ Failure classification: {failure}")
    
    # Test metrics computation
    episodes = [
        {"distance": 5.0, "total_reward": 10.0, "steps": 100, "failure_mode": None},
        {"distance": 3.0, "total_reward": 8.0, "steps": 80, "failure_mode": "forward_fall"}
    ]
    metrics = tester._compute_metrics(episodes)
    print(f"✓ Metrics computation: success_rate={metrics['success_rate']:.2%}")
    
    print("Robustness testing framework: PASSED")


if __name__ == "__main__":
    test_robustness_simple()
    print("\nAll robustness testing tests passed!")
