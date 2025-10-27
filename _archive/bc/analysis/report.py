"""
Automated Evaluation Report Generation

Generate comprehensive markdown reports with analysis and insights.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from passive_walker.bc.evaluate import EvaluationResults, EpisodeMetrics
from passive_walker.bc.visualize import create_comprehensive_plots


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, results: EvaluationResults, 
                       plot_dir: str = "plots") -> str:
        """Generate comprehensive evaluation report."""
        
        # Create plots if they don't exist
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)
            create_comprehensive_plots(results, plot_dir)
        
        # Generate markdown report
        report_content = self._create_report_content(results, plot_dir)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_report_{timestamp}.md"
        report_path = os.path.join(self.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"Report generated: {report_path}")
        return report_path
    
    def _create_report_content(self, results: EvaluationResults, 
                             plot_dir: str) -> str:
        """Create markdown report content."""
        
        content = []
        
        # Header
        content.append("# BC Model Evaluation Report")
        content.append(f"**Generated:** {results.timestamp}")
        content.append(f"**Model:** {os.path.basename(results.model_path)}")
        content.append("")
        
        # Executive Summary
        content.append("## Executive Summary")
        content.append(self._create_executive_summary(results))
        content.append("")
        
        # Configuration
        content.append("## Configuration")
        content.append(self._create_config_section(results))
        content.append("")
        
        # Performance Metrics
        content.append("## Performance Metrics")
        content.append(self._create_performance_section(results))
        content.append("")
        
        # Enhanced Reward Analysis
        content.append("## Enhanced Reward Analysis")
        content.append(self._create_reward_analysis_section(results))
        content.append("")
        
        # Robustness Analysis
        content.append("## Robustness Analysis")
        content.append(self._create_robustness_section(results))
        content.append("")
        
        # Comparison with FSM
        content.append("## Comparison with FSM Baseline")
        content.append(self._create_fsm_comparison_section(results))
        content.append("")
        
        # Failure Analysis
        content.append("## Failure Analysis")
        content.append(self._create_failure_analysis_section(results))
        content.append("")
        
        # Visualizations
        content.append("## Visualizations")
        content.append(self._create_visualization_section(plot_dir))
        content.append("")
        
        # Recommendations
        content.append("## Recommendations")
        content.append(self._create_recommendations_section(results))
        content.append("")
        
        # Appendix
        content.append("## Appendix")
        content.append(self._create_appendix_section(results))
        
        return "\n".join(content)
    
    def _create_executive_summary(self, results: EvaluationResults) -> str:
        """Create executive summary section."""
        summary = []
        
        # Key metrics
        success_rate = results.summary_stats.get('success_rate', 0.0)
        avg_distance = results.summary_stats.get('avg_distance', 0.0)
        avg_reward = results.summary_stats.get('avg_reward', 0.0)
        episode_count = len(results.episodes)
        
        summary.append(f"- **Success Rate:** {success_rate:.1%} ({success_rate * episode_count:.0f}/{episode_count} episodes)")
        summary.append(f"- **Average Distance:** {avg_distance:.2f} meters")
        summary.append(f"- **Average Reward:** {avg_reward:.3f}")
        summary.append(f"- **Total Episodes Evaluated:** {episode_count}")
        
        # Performance assessment
        if success_rate >= 0.8:
            performance_level = "**Excellent**"
            performance_color = "🟢"
        elif success_rate >= 0.6:
            performance_level = "**Good**"
            performance_color = "🟡"
        elif success_rate >= 0.4:
            performance_level = "**Fair**"
            performance_color = "🟠"
        else:
            performance_level = "**Poor**"
            performance_color = "🔴"
        
        summary.append(f"- **Overall Performance:** {performance_color} {performance_level}")
        
        # Key insights
        summary.append("")
        summary.append("### Key Insights")
        
        if success_rate >= 0.8:
            summary.append("- ✅ Model demonstrates robust walking behavior")
            summary.append("- ✅ High success rate indicates good generalization")
        elif success_rate >= 0.6:
            summary.append("- ⚠️ Model shows decent performance with room for improvement")
            summary.append("- ⚠️ Consider additional training or hyperparameter tuning")
        else:
            summary.append("- ❌ Model requires significant improvement")
            summary.append("- ❌ Consider retraining with different data or architecture")
        
        # Enhanced reward insights
        if results.config.get('use_enhanced_rewards', False):
            summary.append("- 🎯 Enhanced reward shaping is active")
            summary.append("- 📊 Detailed reward component analysis available")
        
        return "\n".join(summary)
    
    def _create_config_section(self, results: EvaluationResults) -> str:
        """Create configuration section."""
        config = results.config
        
        config_lines = []
        config_lines.append("| Parameter | Value |")
        config_lines.append("|-----------|-------|")
        
        # Key configuration parameters
        key_params = [
            'backend', 'episodes', 'duration_sec', 'ctrl_hz', 
            'use_enhanced_rewards', 'randomization_profile'
        ]
        
        for param in key_params:
            value = config.get(param, 'N/A')
            config_lines.append(f"| {param} | {value} |")
        
        return "\n".join(config_lines)
    
    def _create_performance_section(self, results: EvaluationResults) -> str:
        """Create performance metrics section."""
        metrics = results.summary_stats
        
        perf_lines = []
        perf_lines.append("### Core Performance Metrics")
        perf_lines.append("")
        perf_lines.append("| Metric | Value | Description |")
        perf_lines.append("|--------|-------|-------------|")
        
        metric_descriptions = {
            'success_rate': 'Percentage of episodes completed successfully',
            'avg_distance': 'Average distance traveled (meters)',
            'avg_duration': 'Average episode duration (seconds)',
            'avg_gait_cycles': 'Average number of gait cycles per episode',
            'avg_reward': 'Average reward per step',
            'avg_energy_efficiency': 'Distance per unit control effort',
            'avg_fsm_imitation_error': 'Average deviation from FSM actions',
            'avg_foot_clearance': 'Average foot clearance height (meters)',
            'avg_velocity_tracking_error': 'Average velocity tracking error',
            'avg_symmetry_error': 'Average left-right symmetry error'
        }
        
        for metric, value in metrics.items():
            description = metric_descriptions.get(metric, 'Performance metric')
            perf_lines.append(f"| {metric} | {value:.3f} | {description} |")
        
        return "\n".join(perf_lines)
    
    def _create_reward_analysis_section(self, results: EvaluationResults) -> str:
        """Create enhanced reward analysis section."""
        if not results.config.get('use_enhanced_rewards', False):
            return "Enhanced rewards are not enabled for this evaluation."
        
        reward_lines = []
        reward_lines.append("### Reward Component Breakdown")
        reward_lines.append("")
        
        # Analyze reward components across episodes
        all_components = {}
        for episode in results.episodes:
            for comp_name, comp_value in episode.reward_components.items():
                if comp_name not in all_components:
                    all_components[comp_name] = []
                all_components[comp_name].append(comp_value)
        
        if all_components:
            reward_lines.append("| Component | Mean | Std | Min | Max | Description |")
            reward_lines.append("|-----------|------|-----|-----|-----|-------------|")
            
            component_descriptions = {
                'r_dx': 'Forward progress reward',
                'r_upright': 'Upright posture bonus',
                'r_velocity': 'Velocity tracking bonus',
                'r_symmetry': 'Left-right symmetry bonus',
                'r_foot_clear': 'Foot clearance bonus',
                'r_ctrl': 'Control effort penalty',
                'r_smooth': 'Smooth motion penalty'
            }
            
            for comp_name, values in all_components.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                description = component_descriptions.get(comp_name, 'Reward component')
                
                reward_lines.append(f"| {comp_name} | {mean_val:.3f} | {std_val:.3f} | {min_val:.3f} | {max_val:.3f} | {description} |")
        
        # Reward insights
        reward_lines.append("")
        reward_lines.append("### Reward Insights")
        
        if 'r_velocity' in all_components:
            vel_mean = np.mean(all_components['r_velocity'])
            if vel_mean > 0.2:
                reward_lines.append("- ✅ Good velocity tracking performance")
            else:
                reward_lines.append("- ⚠️ Velocity tracking could be improved")
        
        if 'r_symmetry' in all_components:
            sym_mean = np.mean(all_components['r_symmetry'])
            if sym_mean > 0.15:
                reward_lines.append("- ✅ Good left-right symmetry")
            else:
                reward_lines.append("- ⚠️ Asymmetry detected in walking pattern")
        
        if 'r_foot_clear' in all_components:
            clear_mean = np.mean(all_components['r_foot_clear'])
            if clear_mean > 0.1:
                reward_lines.append("- ✅ Adequate foot clearance")
            else:
                reward_lines.append("- ⚠️ Low foot clearance may cause tripping")
        
        return "\n".join(reward_lines)
    
    def _create_robustness_section(self, results: EvaluationResults) -> str:
        """Create robustness analysis section."""
        robustness_lines = []
        robustness_lines.append("### Performance Across Physics Conditions")
        robustness_lines.append("")
        
        if results.robustness_matrix:
            robustness_lines.append("| Condition | Success Rate | Avg Distance | Avg Reward | Episodes |")
            robustness_lines.append("|-----------|--------------|--------------|------------|----------|")
            
            for condition, metrics in results.robustness_matrix.items():
                success_rate = metrics.get('success_rate', 0.0)
                avg_distance = metrics.get('avg_distance', 0.0)
                avg_reward = metrics.get('avg_reward', 0.0)
                episode_count = metrics.get('episode_count', 0)
                
                robustness_lines.append(f"| {condition} | {success_rate:.1%} | {avg_distance:.2f}m | {avg_reward:.3f} | {episode_count} |")
            
            # Robustness insights
            robustness_lines.append("")
            robustness_lines.append("### Robustness Insights")
            
            success_rates = [metrics.get('success_rate', 0.0) for metrics in results.robustness_matrix.values()]
            if len(success_rates) > 1:
                robustness_std = np.std(success_rates)
                if robustness_std < 0.1:
                    robustness_lines.append("- ✅ Consistent performance across conditions")
                elif robustness_std < 0.2:
                    robustness_lines.append("- ⚠️ Moderate performance variation across conditions")
                else:
                    robustness_lines.append("- ❌ High performance variation - model not robust")
            
            # Identify best and worst conditions
            best_condition = max(results.robustness_matrix.items(), 
                               key=lambda x: x[1].get('success_rate', 0.0))
            worst_condition = min(results.robustness_matrix.items(), 
                                key=lambda x: x[1].get('success_rate', 0.0))
            
            robustness_lines.append(f"- 🏆 **Best condition:** {best_condition[0]} ({best_condition[1].get('success_rate', 0.0):.1%})")
            robustness_lines.append(f"- 📉 **Worst condition:** {worst_condition[0]} ({worst_condition[1].get('success_rate', 0.0):.1%})")
        
        return "\n".join(robustness_lines)
    
    def _create_fsm_comparison_section(self, results: EvaluationResults) -> str:
        """Create FSM comparison section."""
        comparison_lines = []
        comparison_lines.append("### BC vs FSM Performance Comparison")
        comparison_lines.append("")
        
        fsm_comparison = results.comparison_with_fsm
        
        if fsm_comparison:
            comparison_lines.append("| Metric | BC Model | FSM Baseline | Difference |")
            comparison_lines.append("|--------|----------|--------------|------------|")
            
            bc_success = results.summary_stats.get('success_rate', 0.0)
            fsm_success = fsm_comparison.get('fsm_success_rate', 0.0)
            success_diff = bc_success - fsm_success
            
            bc_distance = results.summary_stats.get('avg_distance', 0.0)
            fsm_distance = fsm_comparison.get('fsm_avg_distance', 0.0)
            distance_diff = bc_distance - fsm_distance
            
            bc_reward = results.summary_stats.get('avg_reward', 0.0)
            fsm_reward = fsm_comparison.get('fsm_avg_reward', 0.0)
            reward_diff = bc_reward - fsm_reward
            
            comparison_lines.append(f"| Success Rate | {bc_success:.1%} | {fsm_success:.1%} | {success_diff:+.1%} |")
            comparison_lines.append(f"| Avg Distance | {bc_distance:.2f}m | {fsm_distance:.2f}m | {distance_diff:+.2f}m |")
            comparison_lines.append(f"| Avg Reward | {bc_reward:.3f} | {fsm_reward:.3f} | {reward_diff:+.3f} |")
            
            # Comparison insights
            comparison_lines.append("")
            comparison_lines.append("### Comparison Insights")
            
            if success_diff > 0.1:
                comparison_lines.append("- 🎉 BC model outperforms FSM baseline!")
            elif success_diff > -0.1:
                comparison_lines.append("- ✅ BC model performance is comparable to FSM")
            else:
                comparison_lines.append("- ⚠️ BC model underperforms compared to FSM")
            
            if distance_diff > 1.0:
                comparison_lines.append("- 🏃 BC model travels significantly farther")
            elif distance_diff < -1.0:
                comparison_lines.append("- 🐌 BC model travels shorter distances")
            
            if reward_diff > 0.1:
                comparison_lines.append("- 💰 BC model achieves higher rewards")
            elif reward_diff < -0.1:
                comparison_lines.append("- 📉 BC model achieves lower rewards")
        
        return "\n".join(comparison_lines)
    
    def _create_failure_analysis_section(self, results: EvaluationResults) -> str:
        """Create failure analysis section."""
        failure_lines = []
        failure_lines.append("### Failure Mode Analysis")
        failure_lines.append("")
        
        # Analyze failed episodes
        failed_episodes = [ep for ep in results.episodes if not ep.success]
        successful_episodes = [ep for ep in results.episodes if ep.success]
        
        if failed_episodes:
            failure_lines.append(f"**Failed Episodes:** {len(failed_episodes)}/{len(results.episodes)}")
            failure_lines.append("")
            
            # Analyze failure patterns
            failed_distances = [ep.distance for ep in failed_episodes]
            failed_durations = [ep.duration for ep in failed_episodes]
            failed_rewards = [ep.avg_reward for ep in failed_episodes]
            
            failure_lines.append("| Metric | Failed Episodes | Successful Episodes |")
            failure_lines.append("|--------|-----------------|---------------------|")
            
            failure_lines.append(f"| Avg Distance | {np.mean(failed_distances):.2f}m | {np.mean([ep.distance for ep in successful_episodes]):.2f}m |")
            failure_lines.append(f"| Avg Duration | {np.mean(failed_durations):.1f}s | {np.mean([ep.duration for ep in successful_episodes]):.1f}s |")
            failure_lines.append(f"| Avg Reward | {np.mean(failed_rewards):.3f} | {np.mean([ep.avg_reward for ep in successful_episodes]):.3f} |")
            
            # Failure insights
            failure_lines.append("")
            failure_lines.append("### Failure Insights")
            
            avg_failure_distance = np.mean(failed_distances)
            avg_success_distance = np.mean([ep.distance for ep in successful_episodes])
            
            if avg_failure_distance < avg_success_distance * 0.5:
                failure_lines.append("- 🚶 Failures occur early in episodes (short distances)")
                failure_lines.append("- 💡 Consider improving initial stability")
            else:
                failure_lines.append("- ⏰ Failures occur later in episodes")
                failure_lines.append("- 💡 Consider improving long-term stability")
            
            # Analyze reward components in failures
            if failed_episodes and failed_episodes[0].reward_components:
                failure_lines.append("- 📊 Analyze reward components in failed episodes")
                failure_lines.append("- 🔍 Look for patterns in control effort or posture")
        else:
            failure_lines.append("🎉 **No failures detected!** All episodes completed successfully.")
        
        return "\n".join(failure_lines)
    
    def _create_visualization_section(self, plot_dir: str) -> str:
        """Create visualization section."""
        viz_lines = []
        viz_lines.append("### Generated Visualizations")
        viz_lines.append("")
        
        # List available plots
        plot_files = []
        if os.path.exists(plot_dir):
            plot_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        
        if plot_files:
            viz_lines.append("The following visualizations have been generated:")
            viz_lines.append("")
            
            plot_descriptions = {
                'episode_comparison': 'Side-by-side comparison of BC vs FSM trajectories',
                'reward_analysis': 'Comprehensive reward component analysis',
                'robustness_matrix': 'Performance across different physics conditions',
                'foot_clearance_analysis': 'Detailed foot clearance analysis'
            }
            
            for plot_file in sorted(plot_files):
                plot_name = plot_file.replace('.png', '')
                description = plot_descriptions.get(plot_name, 'Visualization')
                viz_lines.append(f"- **{plot_file}:** {description}")
        else:
            viz_lines.append("No visualizations available.")
        
        return "\n".join(viz_lines)
    
    def _create_recommendations_section(self, results: EvaluationResults) -> str:
        """Create recommendations section."""
        rec_lines = []
        rec_lines.append("### Recommendations for Improvement")
        rec_lines.append("")
        
        success_rate = results.summary_stats.get('success_rate', 0.0)
        avg_distance = results.summary_stats.get('avg_distance', 0.0)
        avg_reward = results.summary_stats.get('avg_reward', 0.0)
        
        # Success rate recommendations
        if success_rate < 0.6:
            rec_lines.append("#### 🎯 Improve Success Rate")
            rec_lines.append("- Consider retraining with more diverse data")
            rec_lines.append("- Increase training epochs or adjust learning rate")
            rec_lines.append("- Add more physics randomization during training")
            rec_lines.append("- Check for data quality issues")
        
        # Distance recommendations
        if avg_distance < 10.0:
            rec_lines.append("#### 🏃 Improve Distance Performance")
            rec_lines.append("- Focus on forward progress reward component")
            rec_lines.append("- Ensure proper gait cycle training")
            rec_lines.append("- Check for premature termination conditions")
        
        # Reward recommendations
        if avg_reward < 1.0:
            rec_lines.append("#### 💰 Improve Reward Performance")
            rec_lines.append("- Analyze individual reward components")
            rec_lines.append("- Consider reward shaping adjustments")
            rec_lines.append("- Check for reward scaling issues")
        
        # Enhanced reward recommendations
        if results.config.get('use_enhanced_rewards', False):
            rec_lines.append("#### 🎯 Enhanced Reward Optimization")
            
            # Analyze specific reward components
            all_components = {}
            for episode in results.episodes:
                for comp_name, comp_value in episode.reward_components.items():
                    if comp_name not in all_components:
                        all_components[comp_name] = []
                    all_components[comp_name].append(comp_value)
            
            if 'r_velocity' in all_components and np.mean(all_components['r_velocity']) < 0.2:
                rec_lines.append("- Improve velocity tracking (low r_velocity component)")
            
            if 'r_symmetry' in all_components and np.mean(all_components['r_symmetry']) < 0.15:
                rec_lines.append("- Improve left-right symmetry (low r_symmetry component)")
            
            if 'r_foot_clear' in all_components and np.mean(all_components['r_foot_clear']) < 0.1:
                rec_lines.append("- Improve foot clearance (low r_foot_clear component)")
        
        # General recommendations
        rec_lines.append("#### 🔧 General Recommendations")
        rec_lines.append("- Run evaluation on more diverse physics conditions")
        rec_lines.append("- Compare with FSM baseline performance")
        rec_lines.append("- Analyze failure modes in detail")
        rec_lines.append("- Consider ensemble methods for robustness")
        
        return "\n".join(rec_lines)
    
    def _create_appendix_section(self, results: EvaluationResults) -> str:
        """Create appendix section."""
        appendix_lines = []
        appendix_lines.append("### Detailed Episode Data")
        appendix_lines.append("")
        
        appendix_lines.append("| Episode | Success | Distance | Duration | Gait Cycles | Avg Reward |")
        appendix_lines.append("|---------|---------|----------|----------|-------------|------------|")
        
        for i, episode in enumerate(results.episodes):
            appendix_lines.append(f"| {i+1} | {'✅' if episode.success else '❌'} | {episode.distance:.2f}m | {episode.duration:.1f}s | {episode.gait_cycles} | {episode.avg_reward:.3f} |")
        
        appendix_lines.append("")
        appendix_lines.append("### Raw Data Files")
        appendix_lines.append("- `evaluation_results.json`: Complete evaluation data")
        appendix_lines.append("- `episode_*.npz`: Individual episode trajectory data")
        appendix_lines.append("- `*.png`: Generated visualization plots")
        
        return "\n".join(appendix_lines)


def generate_evaluation_report(results: EvaluationResults, 
                             output_dir: str = "reports",
                             plot_dir: str = "plots") -> str:
    """Generate comprehensive evaluation report."""
    generator = ReportGenerator(output_dir)
    return generator.generate_report(results, plot_dir)


if __name__ == "__main__":
    # Example usage
    from passive_walker.bc.evaluate import EvaluationResults, EpisodeMetrics
    
    # Create sample results
    sample_episode = EpisodeMetrics(
        episode_id=0,
        duration=25.0,
        steps=2500,
        success=True,
        distance=15.0,
        gait_cycles=8,
        avg_reward=1.5,
        total_reward=3750.0,
        energy_efficiency=0.1,
        fsm_imitation_error=0.05,
        foot_clearance_avg=0.04,
        velocity_tracking_error=0.1,
        symmetry_error=0.05,
        reward_components={'r_dx': 0.1, 'r_velocity': 0.2, 'r_symmetry': 0.15}
    )
    
    sample_results = EvaluationResults(
        model_path="sample_model.pt",
        config={'use_enhanced_rewards': True, 'episodes': 1},
        episodes=[sample_episode],
        summary_stats={'success_rate': 1.0, 'avg_distance': 15.0},
        robustness_matrix={'nominal': {'success_rate': 1.0}},
        comparison_with_fsm={'fsm_success_rate': 0.95},
        timestamp="2024-01-01 12:00:00"
    )
    
    report_path = generate_evaluation_report(sample_results)
    print(f"Sample report generated: {report_path}")

