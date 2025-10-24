#!/usr/bin/env python3
"""
Advanced Visualization Suite for Passive Walker Models

This module provides comprehensive visualization capabilities:
- 3D trajectory visualization with contact information
- Interactive dashboards for model comparison
- Real-time performance monitoring
- Advanced plotting with custom themes
- Export capabilities for publications
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import time
from datetime import datetime

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


@dataclass
class VisualizationConfig:
    """Configuration for advanced visualization."""
    # Style settings
    style: str = "seaborn-v0_8" if HAS_SEABORN else "default"
    color_palette: str = "viridis"
    figure_size: Tuple[int, int] = (10, 6)  # Reduced default size
    dpi: int = 150  # Reduced DPI for smaller files
    
    # Animation settings
    animation_fps: int = 30
    animation_duration: int = 10  # seconds
    
    # Export settings
    export_formats: List[str] = None
    export_dpi: int = 150  # Reduced DPI for smaller files
    
    # Interactive settings
    enable_interactive: bool = HAS_PLOTLY
    enable_animations: bool = True
    
    def __post_init__(self):
        """Set default values if not provided."""
        if self.export_formats is None:
            self.export_formats = ["png", "pdf", "svg"]


class AdvancedVisualizer:
    """Advanced visualization suite for passive walker analysis."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize advanced visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Setup matplotlib with custom style."""
        plt.style.use(self.config.style)
        
        if HAS_SEABORN:
            sns.set_palette(self.config.color_palette)
        
        # Set default figure size
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['figure.dpi'] = self.config.dpi
        
    def plot_3d_trajectory(self, 
                          trajectory_data: Dict[str, np.ndarray],
                          contact_data: Optional[Dict[str, np.ndarray]] = None,
                          output_path: Optional[str] = None) -> None:
        """Plot 3D trajectory with contact information.
        
        Args:
            trajectory_data: Dictionary with 'x', 'z', 'pitch' arrays
            contact_data: Dictionary with contact information
            output_path: Path to save the plot
        """
        fig = plt.figure(figsize=(10, 8))  # Reduced size
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract trajectory data
        x = trajectory_data['x']
        z = trajectory_data['z']  # Vertical position
        pitch = trajectory_data.get('pitch', np.zeros_like(x))  # Pitch angle
        
        # Plot trajectory
        ax.plot(x, z, pitch, 'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Add contact information if available
        if contact_data is not None:
            left_contact = contact_data.get('left_contact', np.zeros_like(x))
            right_contact = contact_data.get('right_contact', np.zeros_like(x))
            
            # Plot contact points
            left_contact_points = np.where(left_contact > 0.5)[0]
            right_contact_points = np.where(right_contact > 0.5)[0]
            
            if len(left_contact_points) > 0:
                ax.scatter(x[left_contact_points], z[left_contact_points], pitch[left_contact_points],
                          c='red', s=30, alpha=0.8, label='Left Contact')
            
            if len(right_contact_points) > 0:
                ax.scatter(x[right_contact_points], z[right_contact_points], pitch[right_contact_points],
                          c='green', s=30, alpha=0.8, label='Right Contact')
        
        # Add start and end points
        ax.scatter(x[0], z[0], pitch[0], c='green', s=50, marker='o', label='Start')
        ax.scatter(x[-1], z[-1], pitch[-1], c='red', s=50, marker='s', label='End')
        
        # Customize plot
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_zlabel('Pitch Angle (rad)')
        ax.set_title('3D Trajectory Visualization')
        ax.legend()
        
        # Set reasonable limits
        ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
        ax.set_ylim(z.min() - 0.1, z.max() + 0.1)
        ax.set_zlim(pitch.min() - 0.1, pitch.max() + 0.1)
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path, "3d_trajectory")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_gait_analysis(self, 
                          gait_data: Dict[str, np.ndarray],
                          output_path: Optional[str] = None) -> None:
        """Plot comprehensive gait analysis.
        
        Args:
            gait_data: Dictionary with gait metrics
            output_path: Path to save the plot
        """
        # Count available metrics
        available_metrics = [key for key in gait_data.keys() if len(gait_data[key]) > 0]
        
        if not available_metrics:
            print("No gait data available for plotting")
            return
        
        # Create appropriate subplot layout
        n_metrics = len(available_metrics)
        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
        elif n_metrics <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot available metrics
        metric_configs = {
            'step_lengths': ('Step Length Over Time', 'Step Number', 'Step Length (m)', 'b-'),
            'stance_durations': ('Stance Phase Duration', 'Step Number', 'Duration (s)', 'g-'),
            'swing_durations': ('Swing Phase Duration', 'Step Number', 'Duration (s)', 'r-'),
            'double_support_times': ('Double Support Time', 'Step Number', 'Time (s)', 'm-'),
            'walking_speeds': ('Walking Speed', 'Step Number', 'Speed (m/s)', 'c-'),
            'cadence': ('Cadence', 'Step Number', 'Steps/min', 'y-')
        }
        
        for metric in available_metrics:
            if plot_idx >= len(axes):
                break
                
            data = gait_data[metric]
            if len(data) > 0 and metric in metric_configs:
                title, xlabel, ylabel, style = metric_configs[metric]
                
                axes[plot_idx].plot(data, style, linewidth=2)
                axes[plot_idx].set_title(title)
                axes[plot_idx].set_xlabel(xlabel)
                axes[plot_idx].set_ylabel(ylabel)
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path, "gait_analysis")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_model_comparison_dashboard(self, 
                                      comparison_data: Dict[str, Dict[str, np.ndarray]],
                                      output_path: Optional[str] = None) -> None:
        """Plot comprehensive model comparison dashboard.
        
        Args:
            comparison_data: Dictionary mapping model names to their metrics
            output_path: Path to save the plot
        """
        if not comparison_data:
            print("No comparison data available for plotting")
            return
            
        models = list(comparison_data.keys())
        if not models:
            print("No models found in comparison data")
            return
            
        # Get available metrics from first model
        first_model = models[0]
        metrics = [key for key in comparison_data[first_model].keys() 
                  if len(comparison_data[first_model][key]) > 0]
        
        if not metrics:
            print("No metrics with data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))  # Reduced size
        axes = axes.flatten()
        
        # 1. Performance comparison bar chart
        ax1 = axes[0]
        metric_data = {}
        for metric in metrics[:4]:  # Show first 4 metrics
            metric_data[metric] = [np.mean(comparison_data[model][metric]) for model in models]
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric, values) in enumerate(metric_data.items()):
            ax1.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Performance')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot comparison
        ax2 = axes[1]
        box_data = []
        box_labels = []
        
        for model in models:
            for metric in metrics[:2]:  # Show first 2 metrics
                box_data.append(comparison_data[model][metric])
                box_labels.append(f"{model}\n{metric}")
        
        ax2.boxplot(box_data, labels=box_labels)
        ax2.set_title('Performance Distribution')
        ax2.set_ylabel('Values')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance over time
        ax3 = axes[2]
        for model in models:
            if 'success_rate_over_time' in comparison_data[model]:
                ax3.plot(comparison_data[model]['success_rate_over_time'], 
                        label=model, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Radar chart for multiple metrics
        ax4 = axes[3]
        if len(metrics) >= 3:
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for model in models:
                values = [np.mean(comparison_data[model][metric]) for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax4.plot(angles, values, 'o-', linewidth=2, label=model)
                ax4.fill(angles, values, alpha=0.25)
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(metrics)
            ax4.set_title('Multi-Metric Comparison')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path, "model_comparison_dashboard")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_robustness_heatmap(self, 
                              robustness_data: Dict[str, Dict[str, float]],
                              output_path: Optional[str] = None) -> None:
        """Plot robustness testing heatmap.
        
        Args:
            robustness_data: Dictionary mapping conditions to metrics
            output_path: Path to save the plot
        """
        # Extract data for heatmap
        conditions = list(robustness_data.keys())
        metrics = list(robustness_data[conditions[0]].keys())
        
        # Create data matrix
        data_matrix = np.zeros((len(conditions), len(metrics)))
        
        for i, condition in enumerate(conditions):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = robustness_data[condition][metric]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(conditions)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticklabels(conditions)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score')
        
        # Add text annotations
        for i in range(len(conditions)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Robustness Testing Heatmap')
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path, "robustness_heatmap")
        else:
            plt.show()
        
        plt.close(fig)
    
    def create_interactive_dashboard(self, 
                                   evaluation_data: Dict[str, any],
                                   output_path: Optional[str] = None) -> None:
        """Create interactive dashboard using Plotly.
        
        Args:
            evaluation_data: Comprehensive evaluation data
            output_path: Path to save the HTML dashboard
        """
        if not HAS_PLOTLY:
            print("Plotly not available. Creating static dashboard instead.")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Robustness Analysis', 
                           'Failure Distribution', 'Model Comparison'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # 1. Performance metrics bar chart
        if 'performance_metrics' in evaluation_data:
            metrics = evaluation_data['performance_metrics']
            fig.add_trace(
                go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                      name="Performance", showlegend=False),
                row=1, col=1
            )
        
        # 2. Robustness heatmap
        if 'robustness_data' in evaluation_data:
            robustness = evaluation_data['robustness_data']
            conditions = list(robustness.keys())
            metrics = list(robustness[conditions[0]].keys())
            
            z_data = [[robustness[cond][metric] for metric in metrics] for cond in conditions]
            
            fig.add_trace(
                go.Heatmap(z=z_data, x=metrics, y=conditions,
                          colorscale='RdYlBu_r', showscale=True),
                row=1, col=2
            )
        
        # 3. Failure distribution pie chart
        if 'failure_distribution' in evaluation_data:
            failures = evaluation_data['failure_distribution']
            fig.add_trace(
                go.Pie(labels=list(failures.keys()), values=list(failures.values()),
                      name="Failures", showlegend=False),
                row=2, col=1
            )
        
        # 4. Model comparison scatter plot
        if 'model_comparison' in evaluation_data:
            comparison = evaluation_data['model_comparison']
            models = list(comparison.keys())
            
            for model in models:
                if 'success_rate' in comparison[model] and 'distance' in comparison[model]:
                    fig.add_trace(
                        go.Scatter(x=[comparison[model]['success_rate']], 
                                 y=[comparison[model]['distance']],
                                 mode='markers', name=model, showlegend=True),
                        row=2, col=2
                    )
        
        # Update layout
        fig.update_layout(
            title_text="Passive Walker Model Evaluation Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Save interactive dashboard
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path / "interactive_dashboard.html")
            print(f"Interactive dashboard saved: {output_path / 'interactive_dashboard.html'}")
        else:
            fig.show()
    
    def animate_trajectory(self, 
                          trajectory_data: Dict[str, np.ndarray],
                          output_path: Optional[str] = None) -> None:
        """Create animated trajectory visualization.
        
        Args:
            trajectory_data: Dictionary with trajectory data
            output_path: Path to save the animation
        """
        if not self.config.enable_animations:
            print("Animations disabled. Creating static plot instead.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        x = trajectory_data['x']
        y = trajectory_data['y']
        
        # Initialize plot elements
        line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        # Set plot limits
        ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
        ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Animated Trajectory')
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            # Update trajectory line
            line.set_data(x[:frame+1], y[:frame+1])
            
            # Update current position point
            point.set_data([x[frame]], [y[frame]])
            
            return line, point
        
        # Create animation
        frames = len(x)
        interval = 1000 // self.config.animation_fps  # Convert FPS to interval
        
        anim = FuncAnimation(fig, animate, frames=frames, interval=interval, 
                           blit=True, repeat=True)
        
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            anim.save(output_path / "trajectory_animation.gif", writer='pillow', fps=self.config.animation_fps)
            print(f"Animation saved: {output_path / 'trajectory_animation.gif'}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def plot_contact_force_analysis(self, 
                                  contact_data: Dict[str, np.ndarray],
                                  output_path: Optional[str] = None) -> None:
        """Plot detailed contact force analysis.
        
        Args:
            contact_data: Dictionary with contact force data
            output_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # 1. Contact forces over time
        if 'left_force' in contact_data and 'right_force' in contact_data:
            axes[0].plot(contact_data['left_force'], 'r-', linewidth=2, label='Left Foot')
            axes[0].plot(contact_data['right_force'], 'b-', linewidth=2, label='Right Foot')
            axes[0].set_title('Contact Forces Over Time')
            axes[0].set_xlabel('Time Steps')
            axes[0].set_ylabel('Force (N)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 2. Contact duration
        if 'left_contact_duration' in contact_data and 'right_contact_duration' in contact_data:
            axes[1].plot(contact_data['left_contact_duration'], 'r-', linewidth=2, label='Left Foot')
            axes[1].plot(contact_data['right_contact_duration'], 'b-', linewidth=2, label='Right Foot')
            axes[1].set_title('Contact Duration Over Time')
            axes[1].set_xlabel('Time Steps')
            axes[1].set_ylabel('Duration (s)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Force distribution histogram
        if 'left_force' in contact_data:
            axes[2].hist(contact_data['left_force'], bins=50, alpha=0.7, color='red', label='Left Foot')
            axes[2].hist(contact_data['right_force'], bins=50, alpha=0.7, color='blue', label='Right Foot')
            axes[2].set_title('Force Distribution')
            axes[2].set_xlabel('Force (N)')
            axes[2].set_ylabel('Frequency')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # 4. Contact pattern
        if 'left_contact' in contact_data and 'right_contact' in contact_data:
            left_contact = contact_data['left_contact'] > 0.5
            right_contact = contact_data['right_contact'] > 0.5
            
            # Create contact pattern visualization
            time_steps = np.arange(len(left_contact))
            
            axes[3].fill_between(time_steps, 0, 1, where=left_contact, alpha=0.5, color='red', label='Left Contact')
            axes[3].fill_between(time_steps, 1, 2, where=right_contact, alpha=0.5, color='blue', label='Right Contact')
            
            axes[3].set_title('Contact Pattern')
            axes[3].set_xlabel('Time Steps')
            axes[3].set_ylabel('Foot')
            axes[3].set_ylim(0, 2)
            axes[3].set_yticks([0.5, 1.5])
            axes[3].set_yticklabels(['Left', 'Right'])
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            self._save_plot(fig, output_path, "contact_force_analysis")
        else:
            plt.show()
        
        plt.close(fig)
    
    def _save_plot(self, fig, output_path: str, plot_name: str):
        """Save plot in multiple formats."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in self.config.export_formats:
            filename = f"{plot_name}_{timestamp}.{fmt}"
            filepath = output_path / filename
            
            if fmt == "png":
                fig.savefig(filepath, dpi=self.config.export_dpi, bbox_inches='tight')
            elif fmt == "pdf":
                fig.savefig(filepath, bbox_inches='tight')
            elif fmt == "svg":
                fig.savefig(filepath, bbox_inches='tight')
        
        print(f"Plot saved: {output_path / plot_name}_*")
    
    def generate_comprehensive_report(self, 
                                    evaluation_data: Dict[str, any],
                                    output_dir: str = "experiments/outputs/visualization"):
        """Generate comprehensive visualization report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating comprehensive visualization report...")
        
        # Generate different types of visualizations
        if 'trajectory_data' in evaluation_data:
            self.plot_3d_trajectory(
                evaluation_data['trajectory_data'],
                evaluation_data.get('contact_data'),
                str(output_path)
            )
        
        if 'gait_data' in evaluation_data:
            self.plot_gait_analysis(
                evaluation_data['gait_data'],
                str(output_path)
            )
        
        if 'model_comparison' in evaluation_data:
            self.plot_model_comparison_dashboard(
                evaluation_data['model_comparison'],
                str(output_path)
            )
        
        if 'robustness_data' in evaluation_data:
            self.plot_robustness_heatmap(
                evaluation_data['robustness_data'],
                str(output_path)
            )
        
        if 'contact_data' in evaluation_data:
            self.plot_contact_force_analysis(
                evaluation_data['contact_data'],
                str(output_path)
            )
        
        # Create interactive dashboard
        self.create_interactive_dashboard(evaluation_data, str(output_path))
        
        print(f"Comprehensive visualization report generated: {output_path}")


def main():
    """Main function for advanced visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Visualization for Passive Walker Models")
    parser.add_argument("--data-file", type=str, required=True,
                      help="Path to evaluation data file (JSON format)")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/visualization",
                      help="Output directory for visualizations")
    parser.add_argument("--enable-interactive", action="store_true",
                      help="Enable interactive Plotly visualizations")
    parser.add_argument("--enable-animations", action="store_true",
                      help="Enable animated visualizations")
    
    args = parser.parse_args()
    
    # Load evaluation data
    with open(args.data_file, 'r') as f:
        evaluation_data = json.load(f)
    
    # Create visualization configuration
    config = VisualizationConfig(
        enable_interactive=args.enable_interactive,
        enable_animations=args.enable_animations
    )
    
    # Initialize visualizer
    visualizer = AdvancedVisualizer(config)
    
    # Generate comprehensive report
    visualizer.generate_comprehensive_report(evaluation_data, args.output_dir)
    
    print("Advanced visualization completed!")


if __name__ == "__main__":
    main()
