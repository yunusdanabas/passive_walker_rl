#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 3: Evaluation Framework

This test suite validates all Phase 3 components:
- Robustness testing framework
- Distribution shift analysis
- Failure mode detection and analysis
- Integration between evaluation tools
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

from tools.evaluation.robustness_testing import (
    RobustnessConfig, 
    RobustnessResult, 
    RobustnessTester
)
from tools.evaluation.distribution_shift import (
    DistributionShiftConfig,
    DistributionShiftResult,
    DistributionShiftTester
)
from tools.evaluation.failure_analysis import (
    FailureAnalysisConfig,
    FailureEvent,
    FailureAnalysisResult,
    FailureAnalyzer
)
from passive_walker.core.physics_conditions import PhysicsParameter


class TestPhase3RobustnessTesting:
    """Test robustness testing framework."""
    
    def test_robustness_config(self):
        """Test robustness configuration."""
        config = RobustnessConfig()
        
        assert len(config.control_frequencies) == 4
        assert config.control_frequencies == [50.0, 100.0, 150.0, 200.0]
        assert len(config.obs_noise_levels) == 4
        assert len(config.action_noise_levels) == 4
        assert len(config.physics_params) == 4
        assert config.episodes_per_condition == 10
    
    def test_robustness_tester_initialization(self):
        """Test robustness tester initialization."""
        config = RobustnessConfig(episodes_per_condition=2)
        tester = RobustnessTester(config)
        
        assert tester.config == config
        assert tester.results == []
        assert tester.physics_manager is not None
    
    def test_noise_functions(self):
        """Test noise injection functions."""
        tester = RobustnessTester()
        
        # Test observation noise
        obs = np.random.random(17)
        noisy_obs = tester._add_observation_noise(obs, 0.1)
        assert noisy_obs.shape == obs.shape
        
        # Test action noise
        action = np.random.random(3)
        noisy_action = tester._add_action_noise(action, 0.1)
        assert noisy_action.shape == action.shape
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)
        
        # Test dropout
        dropped_obs = tester._apply_dropout(obs, 0.5)
        assert dropped_obs.shape == obs.shape
    
    def test_failure_classification(self):
        """Test failure mode classification."""
        tester = RobustnessTester()
        
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
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        tester = RobustnessTester()
        
        episodes = [
            {"distance": 5.0, "total_reward": 10.0, "steps": 100, "failure_mode": None},
            {"distance": 3.0, "total_reward": 8.0, "steps": 80, "failure_mode": "forward_fall"}
        ]
        
        metrics = tester._compute_metrics(episodes)
        assert metrics["success_rate"] == 0.5
        assert metrics["avg_distance"] == 4.0
        assert metrics["avg_reward"] == 9.0


class TestPhase3DistributionShift:
    """Test distribution shift analysis."""
    
    def test_distribution_shift_config(self):
        """Test distribution shift configuration."""
        config = DistributionShiftConfig()
        
        assert len(config.train_conditions) == 1
        assert len(config.test_conditions) == 7
        assert config.episodes_per_condition == 20
        assert config.kl_divergence_bins == 50
    
    def test_distribution_shift_tester_initialization(self):
        """Test distribution shift tester initialization."""
        config = DistributionShiftConfig(episodes_per_condition=2)
        tester = DistributionShiftTester(config)
        
        assert tester.config == config
        assert tester.results == []
        assert tester.physics_manager is not None
        assert tester.train_observations is None
    
    def test_condition_name_generation(self):
        """Test condition name generation."""
        tester = DistributionShiftTester()
        
        # Single parameter
        condition1 = {"gravity": 0.8}
        name1 = tester._get_condition_name(condition1)
        assert name1 == "gravity_0.80"
        
        # Multiple parameters
        condition2 = {"gravity": 1.0, "mass": 1.2, "friction": 0.7, "damping": 0.5}
        name2 = tester._get_condition_name(condition2)
        assert "mass_1.20" in name2  # Only non-default values included
    
    def test_kl_divergence_computation(self):
        """Test KL divergence computation."""
        tester = DistributionShiftTester()
        
        # Create dummy training observations
        tester.train_observations = np.random.normal(0, 1, (100, 17))
        
        # Create test observations with different distribution
        test_observations = np.random.normal(1, 1, (100, 17))
        
        kl_div = tester._compute_kl_divergence(test_observations)
        assert kl_div > 0  # Should be positive for different distributions
    
    def test_distribution_stats_computation(self):
        """Test distribution statistics computation."""
        tester = DistributionShiftTester()
        
        observations = np.random.normal(0, 1, (100, 17))
        stats = tester._compute_distribution_stats(observations)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats


class TestPhase3FailureAnalysis:
    """Test failure analysis framework."""
    
    def test_failure_analysis_config(self):
        """Test failure analysis configuration."""
        config = FailureAnalysisConfig()
        
        assert config.pitch_fall_threshold == 0.5
        assert config.roll_collapse_threshold == 0.3
        assert config.velocity_stagnation_threshold == 0.1
        assert config.failure_window_size == 50
        assert config.n_clusters == 5
    
    def test_failure_analyzer_initialization(self):
        """Test failure analyzer initialization."""
        config = FailureAnalysisConfig(n_clusters=3)
        analyzer = FailureAnalyzer(config)
        
        assert analyzer.config == config
        assert analyzer.failure_events == []
        assert analyzer.failure_classifier is None
    
    def test_failure_type_classification(self):
        """Test failure type classification."""
        analyzer = FailureAnalyzer()
        
        # Test forward fall
        obs_forward = np.array([1.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure_type = analyzer._classify_failure_type(obs_forward, 0, [obs_forward], [])
        assert failure_type == "forward_fall"
        
        # Test backward fall
        obs_backward = np.array([1.0, 0.0, -0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure_type = analyzer._classify_failure_type(obs_backward, 0, [obs_backward], [])
        assert failure_type == "backward_fall"
        
        # Test stagnation
        obs_stagnant = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        failure_type = analyzer._classify_failure_type(obs_stagnant, 0, [obs_stagnant], [])
        assert failure_type == "stagnation"
    
    def test_failure_event_creation(self):
        """Test failure event creation."""
        failure_event = FailureEvent(
            episode_idx=0,
            step_idx=100,
            failure_type="forward_fall",
            failure_state=np.random.random(17),
            pre_failure_sequence=np.random.random((50, 17)),
            failure_context={"pitch": 0.6, "velocity": 1.0}
        )
        
        assert failure_event.episode_idx == 0
        assert failure_event.step_idx == 100
        assert failure_event.failure_type == "forward_fall"
        assert failure_event.failure_state.shape == (17,)
        assert failure_event.pre_failure_sequence.shape == (50, 17)
        assert failure_event.failure_context["pitch"] == 0.6


class TestPhase3Integration:
    """Integration tests for Phase 3 components."""
    
    def test_evaluation_framework_integration(self):
        """Test integration between evaluation components."""
        # Test that all components can be initialized together
        robustness_config = RobustnessConfig(episodes_per_condition=2)
        robustness_tester = RobustnessTester(robustness_config)
        
        distribution_config = DistributionShiftConfig(episodes_per_condition=2)
        distribution_tester = DistributionShiftTester(distribution_config)
        
        failure_config = FailureAnalysisConfig(n_clusters=3)
        failure_analyzer = FailureAnalyzer(failure_config)
        
        # All components should initialize successfully
        assert robustness_tester is not None
        assert distribution_tester is not None
        assert failure_analyzer is not None
        
        print("✓ Evaluation framework integration: PASSED")
    
    def test_shared_data_structures(self):
        """Test that components use compatible data structures."""
        # Create sample episode data
        episodes = [
            {
                "observations": [np.random.random(17) for _ in range(100)],
                "actions": [np.random.random(3) for _ in range(100)],
                "rewards": [np.random.random() for _ in range(100)],
                "distance": 5.0,
                "total_reward": 10.0,
                "steps": 100,
                "failure_mode": None
            }
        ]
        
        # Test that failure analyzer can process this data
        analyzer = FailureAnalyzer()
        result = analyzer.analyze_failures(episodes)
        
        assert result.total_episodes == 1
        assert isinstance(result.failure_rate, float)
        assert isinstance(result.failure_distribution, dict)
        
        print("✓ Shared data structures: PASSED")
    
    def test_configuration_compatibility(self):
        """Test that configurations are compatible across components."""
        # Test that all configs can be created with similar parameters
        robustness_config = RobustnessConfig(episodes_per_condition=5)
        distribution_config = DistributionShiftConfig(episodes_per_condition=5)
        failure_config = FailureAnalysisConfig()
        
        assert robustness_config.episodes_per_condition == 5
        assert distribution_config.episodes_per_condition == 5
        assert failure_config is not None
        
        print("✓ Configuration compatibility: PASSED")


def test_phase3_simple():
    """Simple test of Phase 3 components."""
    print("\n=== Testing Phase 3 Evaluation Framework ===")
    
    # Test robustness testing
    print("\n--- Robustness Testing ---")
    robustness_config = RobustnessConfig(episodes_per_condition=2)
    robustness_tester = RobustnessTester(robustness_config)
    print(f"✓ RobustnessTester initialized with {len(robustness_tester.config.control_frequencies)} control frequencies")
    
    # Test noise functions
    test_obs = np.random.random(17)
    noisy_obs = robustness_tester._add_observation_noise(test_obs, 0.1)
    print(f"✓ Observation noise injection: {noisy_obs.shape}")
    
    # Test distribution shift
    print("\n--- Distribution Shift Analysis ---")
    distribution_config = DistributionShiftConfig(episodes_per_condition=2)
    distribution_tester = DistributionShiftTester(distribution_config)
    print(f"✓ DistributionShiftTester initialized with {len(distribution_tester.config.test_conditions)} test conditions")
    
    # Test condition name generation
    condition = {"gravity": 0.8}
    name = distribution_tester._get_condition_name(condition)
    print(f"✓ Condition name generation: {name}")
    
    # Test failure analysis
    print("\n--- Failure Analysis ---")
    failure_config = FailureAnalysisConfig(n_clusters=3)
    failure_analyzer = FailureAnalyzer(failure_config)
    print(f"✓ FailureAnalyzer initialized with {failure_analyzer.config.n_clusters} clusters")
    
    # Test failure classification
    obs_forward = np.array([1.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    failure_type = failure_analyzer._classify_failure_type(obs_forward, 0, [obs_forward], [])
    print(f"✓ Failure classification: {failure_type}")
    
    # Test integration
    print("\n--- Integration Testing ---")
    episodes = [
        {
            "observations": [np.random.random(17) for _ in range(50)],
            "actions": [np.random.random(3) for _ in range(50)],
            "rewards": [np.random.random() for _ in range(50)],
            "distance": 5.0,
            "total_reward": 10.0,
            "steps": 50,
            "failure_mode": None
        }
    ]
    
    result = failure_analyzer.analyze_failures(episodes)
    print(f"✓ Failure analysis result: {result.total_episodes} episodes, {result.total_failures} failures")
    
    print("\nPhase 3 evaluation framework: PASSED")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("="*80)
    print("PHASE 3 COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    try:
        # Test robustness testing
        robustness_tests = TestPhase3RobustnessTesting()
        robustness_tests.test_robustness_config()
        robustness_tests.test_robustness_tester_initialization()
        robustness_tests.test_noise_functions()
        robustness_tests.test_failure_classification()
        robustness_tests.test_metrics_computation()
        print("✓ Robustness testing: PASSED")
        
        # Test distribution shift
        distribution_tests = TestPhase3DistributionShift()
        distribution_tests.test_distribution_shift_config()
        distribution_tests.test_distribution_shift_tester_initialization()
        distribution_tests.test_condition_name_generation()
        distribution_tests.test_kl_divergence_computation()
        distribution_tests.test_distribution_stats_computation()
        print("✓ Distribution shift analysis: PASSED")
        
        # Test failure analysis
        failure_tests = TestPhase3FailureAnalysis()
        failure_tests.test_failure_analysis_config()
        failure_tests.test_failure_analyzer_initialization()
        failure_tests.test_failure_type_classification()
        failure_tests.test_failure_event_creation()
        print("✓ Failure analysis: PASSED")
        
        # Test integration
        integration_tests = TestPhase3Integration()
        integration_tests.test_evaluation_framework_integration()
        integration_tests.test_shared_data_structures()
        integration_tests.test_configuration_compatibility()
        print("✓ Integration testing: PASSED")
        
        # Run simple test
        test_phase3_simple()
        
        print("\n" + "="*80)
        print("ALL PHASE 3 TESTS PASSED!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\nPHASE 3 TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
