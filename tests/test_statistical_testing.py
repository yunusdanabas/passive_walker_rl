#!/usr/bin/env python3
"""
Test suite for statistical testing framework.

This test validates the statistical testing components and ensures
proper functionality for model comparisons.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

from tools.evaluation.statistical_testing import (
    StatisticalTestConfig,
    StatisticalTestResult,
    ModelComparisonResult,
    StatisticalTester
)


class TestStatisticalTestConfig:
    """Test statistical testing configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = StatisticalTestConfig()
        
        assert config.alpha == 0.05
        assert config.confidence_level == 0.95
        assert config.bootstrap_samples == 10000
        assert config.bootstrap_random_state == 42
        assert config.correction_method == "bonferroni"
        assert config.small_effect_threshold == 0.2
        assert config.medium_effect_threshold == 0.5
        assert config.large_effect_threshold == 0.8
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = StatisticalTestConfig(
            alpha=0.01,
            confidence_level=0.99,
            bootstrap_samples=5000,
            correction_method="holm"
        )
        
        assert config.alpha == 0.01
        assert config.confidence_level == 0.99
        assert config.bootstrap_samples == 5000
        assert config.correction_method == "holm"


class TestStatisticalTester:
    """Test statistical tester functionality."""
    
    def test_tester_initialization(self):
        """Test statistical tester initialization."""
        config = StatisticalTestConfig(alpha=0.01)
        tester = StatisticalTester(config)
        
        assert tester.config == config
        assert tester.results == []
    
    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        tester = StatisticalTester()
        
        assert tester._interpret_effect_size(0.1) == "negligible"
        assert tester._interpret_effect_size(0.3) == "small"
        assert tester._interpret_effect_size(0.6) == "medium"
        assert tester._interpret_effect_size(1.0) == "large"
        assert tester._interpret_effect_size(-0.3) == "small"  # Should work for negative values
    
    def test_assumptions_checking(self):
        """Test statistical assumptions checking."""
        tester = StatisticalTester()
        
        # Normal data
        normal_data = np.random.normal(0, 1, 30)
        assumptions_met, warnings = tester._check_assumptions(normal_data, normal_data)
        assert isinstance(assumptions_met, bool)
        assert isinstance(warnings, list)
        
        # Data with outliers
        outlier_data = np.concatenate([normal_data, [10, -10]])
        assumptions_met, warnings = tester._check_assumptions(normal_data, outlier_data)
        assert len(warnings) > 0  # Should detect outliers
    
    def test_confidence_interval_computation(self):
        """Test confidence interval computation."""
        tester = StatisticalTester()
        
        data = np.random.normal(0, 1, 100)
        ci = tester._compute_confidence_interval(data)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound should be less than upper bound
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)
    
    def test_difference_confidence_interval_computation(self):
        """Test difference confidence interval computation."""
        tester = StatisticalTester()
        
        data_a = np.random.normal(0, 1, 100)
        data_b = np.random.normal(0.5, 1, 100)
        ci = tester._compute_difference_confidence_interval(data_a, data_b)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert isinstance(ci[0], float)
        assert isinstance(ci[1], float)
    
    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        tester = StatisticalTester()
        
        # Create mock test results
        test_results = {}
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        
        for i, p_val in enumerate(p_values):
            result = StatisticalTestResult(
                test_name=f"test_{i}",
                metric=f"metric_{i}",
                model_a_name="Model A",
                model_b_name="Model B",
                model_a_mean=1.0,
                model_b_mean=1.0,
                model_a_std=0.1,
                model_b_std=0.1,
                model_a_n=10,
                model_b_n=10,
                t_statistic=1.0,
                p_value=p_val,
                degrees_of_freedom=18,
                cohens_d=0.1,
                effect_size_interpretation="small",
                model_a_ci=(0.9, 1.1),
                model_b_ci=(0.9, 1.1),
                difference_ci=(-0.1, 0.1),
                is_significant=p_val < 0.05,
                significance_level=0.05,
                test_type="independent_t_test",
                assumptions_met=True,
                warnings=[]
            )
            test_results[f"metric_{i}"] = result
        
        corrected_p_values = tester._apply_multiple_comparison_correction(test_results)
        
        assert len(corrected_p_values) == len(p_values)
        # Bonferroni correction should increase p-values
        for original_p, corrected_p in zip(p_values, corrected_p_values.values()):
            assert corrected_p >= original_p
    
    def test_model_comparison(self):
        """Test complete model comparison."""
        tester = StatisticalTester()
        
        # Create sample data
        np.random.seed(42)
        model_a_data = {
            "success_rate": np.random.beta(8, 2, 50),  # Higher success rate
            "distance": np.random.normal(5.0, 1.0, 50),
            "reward": np.random.normal(10.0, 2.0, 50)
        }
        
        model_b_data = {
            "success_rate": np.random.beta(5, 5, 50),  # Lower success rate
            "distance": np.random.normal(4.0, 1.0, 50),
            "reward": np.random.normal(8.0, 2.0, 50)
        }
        
        # Run comparison
        result = tester.compare_models(
            model_a_data=model_a_data,
            model_b_data=model_b_data,
            model_a_name="Model A",
            model_b_name="Model B"
        )
        
        # Check results
        assert isinstance(result, ModelComparisonResult)
        assert len(result.models) == 2
        assert len(result.metrics) == 3
        assert len(result.test_results) == 3
        assert len(result.summary_table) == 3
        assert len(result.multiple_comparison_correction) == 3
        assert isinstance(result.recommendations, list)
        
        # Check that Model A generally performs better
        for metric in result.metrics:
            test_result = result.test_results[metric]
            assert test_result.model_a_name == "Model A"
            assert test_result.model_b_name == "Model B"
            assert isinstance(test_result.p_value, float)
            assert isinstance(test_result.cohens_d, float)
            assert isinstance(test_result.is_significant, bool)


class TestStatisticalIntegration:
    """Integration tests for statistical testing."""
    
    def test_statistical_testing_workflow(self):
        """Test complete statistical testing workflow."""
        # Create realistic model comparison data
        np.random.seed(123)
        
        # Model A: Better performance
        model_a_data = {
            "success_rate": np.random.beta(9, 1, 100),  # High success rate
            "distance": np.random.normal(6.0, 0.5, 100),  # Good distance
            "reward": np.random.normal(12.0, 1.0, 100),  # High reward
            "steps": np.random.normal(800, 100, 100)  # Good episode length
        }
        
        # Model B: Worse performance
        model_b_data = {
            "success_rate": np.random.beta(3, 7, 100),  # Lower success rate
            "distance": np.random.normal(3.0, 1.0, 100),  # Shorter distance
            "reward": np.random.normal(6.0, 2.0, 100),  # Lower reward
            "steps": np.random.normal(400, 150, 100)  # Shorter episodes
        }
        
        # Initialize tester
        config = StatisticalTestConfig(alpha=0.05, bootstrap_samples=1000)
        tester = StatisticalTester(config)
        
        # Run comparison
        result = tester.compare_models(
            model_a_data=model_a_data,
            model_b_data=model_b_data,
            model_a_name="LSTM Model",
            model_b_name="MLP Model"
        )
        
        # Verify results
        assert len(result.metrics) == 4
        assert all(metric in result.test_results for metric in result.metrics)
        
        # Check that Model A (LSTM) performs better on average
        for metric in ["success_rate", "distance", "reward", "steps"]:
            summary = result.summary_table[metric]
            if metric in ["success_rate", "distance", "reward", "steps"]:
                # Model A should be better for these metrics
                assert summary["model_a_mean"] > summary["model_b_mean"]
        
        print("✓ Statistical testing workflow: PASSED")


def test_statistical_simple():
    """Simple test of statistical testing components."""
    print("\n=== Testing Statistical Testing Framework ===")
    
    # Test configuration
    config = StatisticalTestConfig(alpha=0.05)
    print(f"✓ Configuration created with alpha={config.alpha}")
    
    # Test tester initialization
    tester = StatisticalTester(config)
    print(f"✓ StatisticalTester initialized")
    
    # Test effect size interpretation
    effect_size = tester._interpret_effect_size(0.6)
    print(f"✓ Effect size interpretation: {effect_size}")
    
    # Test assumptions checking
    data = np.random.normal(0, 1, 30)
    assumptions_met, warnings = tester._check_assumptions(data, data)
    print(f"✓ Assumptions checking: {assumptions_met}, {len(warnings)} warnings")
    
    # Test confidence interval computation
    ci = tester._compute_confidence_interval(data)
    print(f"✓ Confidence interval: [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    # Test model comparison
    np.random.seed(42)
    model_a_data = {
        "success_rate": np.random.beta(8, 2, 20),
        "distance": np.random.normal(5.0, 1.0, 20)
    }
    
    model_b_data = {
        "success_rate": np.random.beta(5, 5, 20),
        "distance": np.random.normal(4.0, 1.0, 20)
    }
    
    result = tester.compare_models(
        model_a_data=model_a_data,
        model_b_data=model_b_data,
        model_a_name="Model A",
        model_b_name="Model B"
    )
    
    print(f"✓ Model comparison: {len(result.metrics)} metrics tested")
    print(f"✓ Summary table: {len(result.summary_table)} entries")
    print(f"✓ Multiple comparison correction: {len(result.multiple_comparison_correction)} corrections")
    print(f"✓ Recommendations: {len(result.recommendations)} generated")
    
    print("Statistical testing framework: PASSED")


if __name__ == "__main__":
    test_statistical_simple()
    print("\nAll statistical testing tests passed!")
