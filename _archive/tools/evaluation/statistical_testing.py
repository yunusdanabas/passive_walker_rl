#!/usr/bin/env python3
"""
Statistical Testing Framework for Passive Walker Models

This module implements rigorous statistical analysis for model comparisons:
- Paired t-tests for model A vs model B
- Bootstrap confidence intervals for metrics
- Effect size computation (Cohen's d)
- Multiple comparison correction (Bonferroni)
- Statistical significance testing
- Sample size recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
from scipy import stats
from scipy.stats import bootstrap
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical testing."""
    # Significance level
    alpha: float = 0.05
    
    # Confidence interval level
    confidence_level: float = 0.95
    
    # Bootstrap parameters
    bootstrap_samples: int = 10000
    bootstrap_random_state: int = 42
    
    # Multiple comparison correction
    correction_method: str = "bonferroni"  # "bonferroni", "holm", "fdr"
    
    # Effect size thresholds
    small_effect_threshold: float = 0.2
    medium_effect_threshold: float = 0.5
    large_effect_threshold: float = 0.8


@dataclass
class StatisticalTestResult:
    """Results from statistical testing."""
    test_name: str
    metric: str
    model_a_name: str
    model_b_name: str
    
    # Basic statistics
    model_a_mean: float
    model_b_mean: float
    model_a_std: float
    model_b_std: float
    model_a_n: int
    model_b_n: int
    
    # Test statistics
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    
    # Effect size
    cohens_d: float
    effect_size_interpretation: str
    
    # Confidence intervals
    model_a_ci: Tuple[float, float]
    model_b_ci: Tuple[float, float]
    difference_ci: Tuple[float, float]
    
    # Significance
    is_significant: bool
    significance_level: float
    
    # Additional info
    test_type: str
    assumptions_met: bool
    warnings: List[str]


@dataclass
class ModelComparisonResult:
    """Results from comprehensive model comparison."""
    models: List[str]
    metrics: List[str]
    test_results: Dict[str, StatisticalTestResult]
    summary_table: Dict[str, Dict[str, float]]
    multiple_comparison_correction: Dict[str, float]
    recommendations: List[str]


class StatisticalTester:
    """Statistical testing framework for model comparisons."""
    
    def __init__(self, config: Optional[StatisticalTestConfig] = None):
        """Initialize statistical tester.
        
        Args:
            config: Statistical testing configuration
        """
        self.config = config or StatisticalTestConfig()
        self.results: List[StatisticalTestResult] = []
        
    def compare_models(self, 
                      model_a_data: Dict[str, np.ndarray],
                      model_b_data: Dict[str, np.ndarray],
                      model_a_name: str = "Model A",
                      model_b_name: str = "Model B",
                      metrics: Optional[List[str]] = None) -> ModelComparisonResult:
        """Compare two models across multiple metrics.
        
        Args:
            model_a_data: Dictionary mapping metric names to arrays of values
            model_b_data: Dictionary mapping metric names to arrays of values
            model_a_name: Name of first model
            model_b_name: Name of second model
            metrics: List of metrics to compare (None for all)
            
        Returns:
            Comprehensive model comparison results
        """
        print(f"Comparing {model_a_name} vs {model_b_name}")
        
        if metrics is None:
            metrics = list(set(model_a_data.keys()) & set(model_b_data.keys()))
        
        test_results = {}
        
        for metric in metrics:
            print(f"  Testing metric: {metric}")
            
            if metric not in model_a_data or metric not in model_b_data:
                print(f"    Warning: Metric {metric} not found in both models")
                continue
            
            data_a = model_a_data[metric]
            data_b = model_b_data[metric]
            
            # Perform statistical test
            result = self._perform_statistical_test(
                data_a, data_b, metric, model_a_name, model_b_name
            )
            
            test_results[metric] = result
            self.results.append(result)
        
        # Apply multiple comparison correction
        corrected_p_values = self._apply_multiple_comparison_correction(test_results)
        
        # Generate summary table
        summary_table = self._generate_summary_table(test_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(test_results, corrected_p_values)
        
        return ModelComparisonResult(
            models=[model_a_name, model_b_name],
            metrics=metrics,
            test_results=test_results,
            summary_table=summary_table,
            multiple_comparison_correction=corrected_p_values,
            recommendations=recommendations
        )
    
    def _perform_statistical_test(self, 
                                data_a: np.ndarray,
                                data_b: np.ndarray,
                                metric: str,
                                model_a_name: str,
                                model_b_name: str) -> StatisticalTestResult:
        """Perform statistical test between two datasets."""
        # Basic statistics
        mean_a = np.mean(data_a)
        mean_b = np.mean(data_b)
        std_a = np.std(data_a, ddof=1)
        std_b = np.std(data_b, ddof=1)
        n_a = len(data_a)
        n_b = len(data_b)
        
        # Check assumptions
        assumptions_met, warnings = self._check_assumptions(data_a, data_b)
        
        # Perform paired t-test (assuming paired data)
        if len(data_a) == len(data_b):
            # Paired t-test
            differences = data_a - data_b
            t_stat, p_value = stats.ttest_rel(data_a, data_b)
            df = n_a - 1
            test_type = "paired_t_test"
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(data_a, data_b)
            df = n_a + n_b - 2
            test_type = "independent_t_test"
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        cohens_d = (mean_a - mean_b) / pooled_std
        effect_size_interpretation = self._interpret_effect_size(cohens_d)
        
        # Compute confidence intervals
        ci_a = self._compute_confidence_interval(data_a)
        ci_b = self._compute_confidence_interval(data_b)
        ci_diff = self._compute_difference_confidence_interval(data_a, data_b)
        
        # Determine significance
        is_significant = p_value < self.config.alpha
        
        return StatisticalTestResult(
            test_name=f"{model_a_name}_vs_{model_b_name}_{metric}",
            metric=metric,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            model_a_mean=mean_a,
            model_b_mean=mean_b,
            model_a_std=std_a,
            model_b_std=std_b,
            model_a_n=n_a,
            model_b_n=n_b,
            t_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            cohens_d=cohens_d,
            effect_size_interpretation=effect_size_interpretation,
            model_a_ci=ci_a,
            model_b_ci=ci_b,
            difference_ci=ci_diff,
            is_significant=is_significant,
            significance_level=self.config.alpha,
            test_type=test_type,
            assumptions_met=assumptions_met,
            warnings=warnings
        )
    
    def _check_assumptions(self, data_a: np.ndarray, data_b: np.ndarray) -> Tuple[bool, List[str]]:
        """Check statistical test assumptions."""
        warnings = []
        assumptions_met = True
        
        # Check normality (Shapiro-Wilk test for small samples)
        if len(data_a) <= 50:
            _, p_a = stats.shapiro(data_a)
            _, p_b = stats.shapiro(data_b)
            
            if p_a < 0.05:
                warnings.append(f"Model A data may not be normally distributed (p={p_a:.3f})")
                assumptions_met = False
            
            if p_b < 0.05:
                warnings.append(f"Model B data may not be normally distributed (p={p_b:.3f})")
                assumptions_met = False
        
        # Check for outliers (using IQR method)
        for i, data in enumerate([data_a, data_b]):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = np.sum((data < lower_bound) | (data > upper_bound))
            if outliers > 0:
                warnings.append(f"Model {'A' if i == 0 else 'B'} has {outliers} outliers")
        
        return assumptions_met, warnings
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        
        if abs_d < self.config.small_effect_threshold:
            return "negligible"
        elif abs_d < self.config.medium_effect_threshold:
            return "small"
        elif abs_d < self.config.large_effect_threshold:
            return "medium"
        else:
            return "large"
    
    def _compute_confidence_interval(self, data: np.ndarray) -> Tuple[float, float]:
        """Compute confidence interval using bootstrap."""
        try:
            # Use scipy bootstrap
            bootstrap_result = bootstrap(
                (data,), 
                np.mean, 
                n_resamples=self.config.bootstrap_samples,
                confidence_level=self.config.confidence_level,
                random_state=self.config.bootstrap_random_state
            )
            return bootstrap_result.confidence_interval
        except Exception:
            # Fallback to t-distribution
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            n = len(data)
            se = std / np.sqrt(n)
            t_critical = stats.t.ppf(1 - (1 - self.config.confidence_level) / 2, n - 1)
            margin = t_critical * se
            return (mean - margin, mean + margin)
    
    def _compute_difference_confidence_interval(self, data_a: np.ndarray, data_b: np.ndarray) -> Tuple[float, float]:
        """Compute confidence interval for the difference."""
        try:
            # Bootstrap difference
            def difference_statistic(data):
                return np.mean(data[0]) - np.mean(data[1])
            
            bootstrap_result = bootstrap(
                (data_a, data_b), 
                difference_statistic, 
                n_resamples=self.config.bootstrap_samples,
                confidence_level=self.config.confidence_level,
                random_state=self.config.bootstrap_random_state
            )
            return bootstrap_result.confidence_interval
        except Exception:
            # Fallback to t-distribution
            diff_mean = np.mean(data_a) - np.mean(data_b)
            diff_std = np.sqrt(np.var(data_a, ddof=1) / len(data_a) + np.var(data_b, ddof=1) / len(data_b))
            n_eff = min(len(data_a), len(data_b))
            t_critical = stats.t.ppf(1 - (1 - self.config.confidence_level) / 2, n_eff - 1)
            margin = t_critical * diff_std
            return (diff_mean - margin, diff_mean + margin)
    
    def _apply_multiple_comparison_correction(self, test_results: Dict[str, StatisticalTestResult]) -> Dict[str, float]:
        """Apply multiple comparison correction."""
        p_values = [result.p_value for result in test_results.values()]
        metric_names = list(test_results.keys())
        
        if self.config.correction_method == "bonferroni":
            corrected_p_values = [min(1.0, p * len(p_values)) for p in p_values]
        elif self.config.correction_method == "holm":
            # Holm-Bonferroni correction
            sorted_indices = np.argsort(p_values)
            corrected_p_values = [0.0] * len(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p_values[idx] = min(1.0, p_values[idx] * (len(p_values) - i))
        elif self.config.correction_method == "fdr":
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            corrected_p_values = [0.0] * len(p_values)
            for i, idx in enumerate(sorted_indices):
                corrected_p_values[idx] = min(1.0, p_values[idx] * len(p_values) / (i + 1))
        else:
            corrected_p_values = p_values
        
        return dict(zip(metric_names, corrected_p_values))
    
    def _generate_summary_table(self, test_results: Dict[str, StatisticalTestResult]) -> Dict[str, Dict[str, float]]:
        """Generate summary table of results."""
        summary = {}
        
        for metric, result in test_results.items():
            summary[metric] = {
                "model_a_mean": result.model_a_mean,
                "model_b_mean": result.model_b_mean,
                "difference": result.model_a_mean - result.model_b_mean,
                "p_value": result.p_value,
                "cohens_d": result.cohens_d,
                "effect_size": result.effect_size_interpretation,
                "is_significant": result.is_significant
            }
        
        return summary
    
    def _generate_recommendations(self, 
                                test_results: Dict[str, StatisticalTestResult],
                                corrected_p_values: Dict[str, float]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Check sample sizes
        for result in test_results.values():
            if result.model_a_n < 30 or result.model_b_n < 30:
                recommendations.append(
                    f"Consider increasing sample size for {result.metric} "
                    f"(current: {result.model_a_n}, {result.model_b_n})"
                )
        
        # Check effect sizes
        large_effects = [r for r in test_results.values() if r.effect_size_interpretation == "large"]
        if large_effects:
            recommendations.append(
                f"Large effect sizes detected in {len(large_effects)} metrics - "
                "results are practically significant"
            )
        
        # Check significance after correction
        significant_after_correction = sum(1 for p in corrected_p_values.values() if p < self.config.alpha)
        if significant_after_correction == 0:
            recommendations.append(
                "No significant differences after multiple comparison correction - "
                "consider increasing sample size or using more sensitive tests"
            )
        
        # Check assumptions
        failed_assumptions = [r for r in test_results.values() if not r.assumptions_met]
        if failed_assumptions:
            recommendations.append(
                f"Statistical assumptions violated in {len(failed_assumptions)} tests - "
                "consider non-parametric alternatives"
            )
        
        return recommendations
    
    def generate_statistical_report(self, 
                                  comparison_result: ModelComparisonResult,
                                  output_dir: str = "experiments/outputs/statistical_testing"):
        """Generate comprehensive statistical report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate summary table plot
        self._plot_summary_table(comparison_result, output_path)
        
        # Generate effect size plot
        self._plot_effect_sizes(comparison_result, output_path)
        
        # Generate confidence interval plot
        self._plot_confidence_intervals(comparison_result, output_path)
        
        # Generate text report
        self._generate_text_report(comparison_result, output_path)
        
        print(f"Statistical report generated: {output_path}")
    
    def _plot_summary_table(self, result: ModelComparisonResult, output_path: Path):
        """Generate summary table visualization."""
        metrics = result.metrics
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=(12, max(6, n_metrics * 0.5)))
        
        # Create data for plotting
        model_a_means = [result.summary_table[m]["model_a_mean"] for m in metrics]
        model_b_means = [result.summary_table[m]["model_b_mean"] for m in metrics]
        p_values = [result.summary_table[m]["p_value"] for m in metrics]
        
        x = np.arange(n_metrics)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, model_a_means, width, label=result.models[0], alpha=0.8)
        bars2 = ax.bar(x + width/2, model_b_means, width, label=result.models[1], alpha=0.8)
        
        # Add significance annotations
        for i, (p_val, corrected_p) in enumerate(zip(p_values, result.multiple_comparison_correction.values())):
            y_pos = max(model_a_means[i], model_b_means[i]) + 0.1
            
            if corrected_p < 0.001:
                sig_text = "***"
            elif corrected_p < 0.01:
                sig_text = "**"
            elif corrected_p < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            
            ax.text(i, y_pos, sig_text, ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Model Comparison Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "summary_table.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_effect_sizes(self, result: ModelComparisonResult, output_path: Path):
        """Generate effect size visualization."""
        metrics = result.metrics
        cohens_d_values = [result.summary_table[m]["cohens_d"] for m in metrics]
        effect_sizes = [result.summary_table[m]["effect_size"] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color by effect size
        colors = []
        for effect_size in effect_sizes:
            if effect_size == "negligible":
                colors.append('lightgray')
            elif effect_size == "small":
                colors.append('yellow')
            elif effect_size == "medium":
                colors.append('orange')
            else:  # large
                colors.append('red')
        
        bars = ax.bar(range(len(metrics)), cohens_d_values, color=colors, alpha=0.7)
        
        # Add effect size thresholds
        ax.axhline(y=self.config.small_effect_threshold, color='green', linestyle='--', alpha=0.5, label='Small effect')
        ax.axhline(y=self.config.medium_effect_threshold, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axhline(y=self.config.large_effect_threshold, color='red', linestyle='--', alpha=0.5, label='Large effect')
        ax.axhline(y=-self.config.small_effect_threshold, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-self.config.medium_effect_threshold, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(y=-self.config.large_effect_threshold, color='red', linestyle='--', alpha=0.5)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel("Cohen's d")
        ax.set_title('Effect Sizes (Cohen\'s d)')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_intervals(self, result: ModelComparisonResult, output_path: Path):
        """Generate confidence interval visualization."""
        metrics = result.metrics
        n_metrics = len(metrics)
        
        fig, ax = plt.subplots(figsize=(12, max(6, n_metrics * 0.5)))
        
        y_positions = np.arange(n_metrics)
        
        for i, metric in enumerate(metrics):
            test_result = result.test_results[metric]
            
            # Plot confidence intervals
            ax.errorbar(test_result.model_a_mean, i - 0.2, 
                       xerr=[[test_result.model_a_mean - test_result.model_a_ci[0]], 
                             [test_result.model_a_ci[1] - test_result.model_a_mean]], 
                       fmt='o', label=result.models[0] if i == 0 else "", capsize=5)
            
            ax.errorbar(test_result.model_b_mean, i + 0.2, 
                       xerr=[[test_result.model_b_mean - test_result.model_b_ci[0]], 
                             [test_result.model_b_ci[1] - test_result.model_b_mean]], 
                       fmt='s', label=result.models[1] if i == 0 else "", capsize=5)
        
        ax.set_xlabel('Values')
        ax.set_ylabel('Metrics')
        ax.set_title('Confidence Intervals')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "confidence_intervals.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, result: ModelComparisonResult, output_path: Path):
        """Generate text summary report."""
        report_path = output_path / "statistical_test_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("STATISTICAL TESTING SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Models compared: {result.models[0]} vs {result.models[1]}\n")
            f.write(f"Metrics tested: {len(result.metrics)}\n")
            f.write(f"Significance level: {self.config.alpha}\n")
            f.write(f"Confidence level: {self.config.confidence_level}\n")
            f.write(f"Multiple comparison correction: {self.config.correction_method}\n\n")
            
            f.write("SUMMARY TABLE\n")
            f.write("-" * 20 + "\n")
            cohens_d_header = "Cohen's d"
            f.write(f"{'Metric':<15} {'Model A':<10} {'Model B':<10} {'Diff':<8} {'p-value':<8} {cohens_d_header:<10} {'Effect':<10} {'Sig':<5}\n")
            f.write("-" * 80 + "\n")
            
            for metric in result.metrics:
                summary = result.summary_table[metric]
                corrected_p = result.multiple_comparison_correction[metric]
                sig = "Yes" if corrected_p < self.config.alpha else "No"
                
                f.write(f"{metric:<15} {summary['model_a_mean']:<10.3f} {summary['model_b_mean']:<10.3f} "
                       f"{summary['difference']:<8.3f} {corrected_p:<8.3f} {summary['cohens_d']:<10.3f} "
                       f"{summary['effect_size']:<10} {sig:<5}\n")
            
            f.write("\nCORRECTED P-VALUES\n")
            f.write("-" * 20 + "\n")
            for metric, corrected_p in result.multiple_comparison_correction.items():
                f.write(f"{metric}: {corrected_p:.6f}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(result.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\nDETAILED RESULTS\n")
            f.write("-" * 20 + "\n")
            for metric, test_result in result.test_results.items():
                f.write(f"\n{metric}:\n")
                f.write(f"  Test type: {test_result.test_type}\n")
                f.write(f"  t-statistic: {test_result.t_statistic:.4f}\n")
                f.write(f"  Degrees of freedom: {test_result.degrees_of_freedom}\n")
                f.write(f"  p-value: {test_result.p_value:.6f}\n")
                f.write(f"  Cohen's d: {test_result.cohens_d:.4f} ({test_result.effect_size_interpretation})\n")
                f.write(f"  Model A CI: [{test_result.model_a_ci[0]:.3f}, {test_result.model_a_ci[1]:.3f}]\n")
                f.write(f"  Model B CI: [{test_result.model_b_ci[0]:.3f}, {test_result.model_b_ci[1]:.3f}]\n")
                f.write(f"  Difference CI: [{test_result.difference_ci[0]:.3f}, {test_result.difference_ci[1]:.3f}]\n")
                f.write(f"  Assumptions met: {test_result.assumptions_met}\n")
                if test_result.warnings:
                    f.write(f"  Warnings: {', '.join(test_result.warnings)}\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    """Main function for statistical testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical Testing for Passive Walker Models")
    parser.add_argument("--model-a-data", type=str, required=True,
                      help="Path to Model A data file (JSON format)")
    parser.add_argument("--model-b-data", type=str, required=True,
                      help="Path to Model B data file (JSON format)")
    parser.add_argument("--model-a-name", type=str, default="Model A",
                      help="Name of Model A")
    parser.add_argument("--model-b-name", type=str, default="Model B",
                      help="Name of Model B")
    parser.add_argument("--alpha", type=float, default=0.05,
                      help="Significance level")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/statistical_testing",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load model data
    with open(args.model_a_data, 'r') as f:
        model_a_data = json.load(f)
    
    with open(args.model_b_data, 'r') as f:
        model_b_data = json.load(f)
    
    # Convert to numpy arrays
    for key in model_a_data:
        model_a_data[key] = np.array(model_a_data[key])
    
    for key in model_b_data:
        model_b_data[key] = np.array(model_b_data[key])
    
    # Create statistical testing configuration
    config = StatisticalTestConfig(alpha=args.alpha)
    
    # Initialize tester
    tester = StatisticalTester(config)
    
    # Run comparison
    result = tester.compare_models(
        model_a_data=model_a_data,
        model_b_data=model_b_data,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name
    )
    
    # Generate report
    tester.generate_statistical_report(result, args.output_dir)
    
    print("Statistical testing completed!")


if __name__ == "__main__":
    main()
