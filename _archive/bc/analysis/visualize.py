"""
Trajectory Visualization and Comparison Tools

Comprehensive plotting tools for BC model analysis and comparison.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path

# Optional seaborn import
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Set style
plt.style.use('default')  # Use default instead of seaborn


class TrajectoryVisualizer:
    """Comprehensive trajectory visualization tools."""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_episode_comparison(self, bc_data: Dict, fsm_data: Dict, 
                              episode_id: int = 0, save: bool = True) -> str:
        """Plot side-by-side comparison of BC vs FSM episode."""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time axis
        time_bc = np.linspace(0, len(bc_data['observations']) * 0.01, len(bc_data['observations']))
        time_fsm = np.linspace(0, len(fsm_data['observations']) * 0.01, len(fsm_data['observations']))
        
        # 1. Joint angle trajectories
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_joint_trajectories(ax1, bc_data, time_bc, "BC Model", color='blue')
        ax1.set_title("BC Joint Angles")
        ax1.set_ylabel("Angle (rad)")
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_joint_trajectories(ax2, fsm_data, time_fsm, "FSM", color='red')
        ax2.set_title("FSM Joint Angles")
        ax2.set_ylabel("Angle (rad)")
        
        # 2. Control signals
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_control_signals(ax3, bc_data, time_bc, "BC Model", color='blue')
        ax3.set_title("BC Control Signals")
        ax3.set_ylabel("Action")
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_control_signals(ax4, fsm_data, time_fsm, "FSM", color='red')
        ax4.set_title("FSM Control Signals")
        ax4.set_ylabel("Action")
        
        # 3. Reward components
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_reward_components(ax5, bc_data, time_bc)
        ax5.set_title("BC Reward Components Over Time")
        ax5.set_ylabel("Reward")
        
        # 4. Phase portraits
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_phase_portrait(ax6, bc_data, "BC Model", color='blue')
        ax6.set_title("BC Phase Portrait")
        
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_phase_portrait(ax7, fsm_data, "FSM", color='red')
        ax7.set_title("FSM Phase Portrait")
        
        plt.suptitle(f"Episode {episode_id} Comparison: BC vs FSM", fontsize=16)
        
        if save:
            filename = f"episode_{episode_id:03d}_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_reward_analysis(self, episodes: List[Dict], save: bool = True) -> str:
        """Plot comprehensive reward analysis across episodes."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Extract reward components
        reward_components = {}
        episode_rewards = []
        episode_distances = []
        
        for ep in episodes:
            episode_rewards.append(ep['total_reward'])
            episode_distances.append(ep['distance'])
            
            for comp_name, comp_value in ep['reward_components'].items():
                if comp_name not in reward_components:
                    reward_components[comp_name] = []
                reward_components[comp_name].append(comp_value)
        
        # 1. Total reward distribution
        axes[0].hist(episode_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].set_title("Total Reward Distribution")
        axes[0].set_xlabel("Total Reward")
        axes[0].set_ylabel("Frequency")
        axes[0].axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
        axes[0].legend()
        
        # 2. Reward vs Distance
        axes[1].scatter(episode_distances, episode_rewards, alpha=0.7, color='green')
        axes[1].set_title("Reward vs Distance")
        axes[1].set_xlabel("Distance (m)")
        axes[1].set_ylabel("Total Reward")
        
        # Add trend line
        z = np.polyfit(episode_distances, episode_rewards, 1)
        p = np.poly1d(z)
        axes[1].plot(episode_distances, p(episode_distances), "r--", alpha=0.8)
        
        # 3. Reward components box plot
        if reward_components:
            comp_names = list(reward_components.keys())
            comp_values = [reward_components[name] for name in comp_names]
            
            bp = axes[2].boxplot(comp_values, labels=comp_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            axes[2].set_title("Reward Components Distribution")
            axes[2].set_ylabel("Reward Value")
            axes[2].tick_params(axis='x', rotation=45)
        
        # 4. Reward components correlation heatmap
        if len(reward_components) > 1:
            comp_matrix = np.array([reward_components[name] for name in comp_names]).T
            corr_matrix = np.corrcoef(comp_matrix.T)
            
            im = axes[3].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[3].set_xticks(range(len(comp_names)))
            axes[3].set_yticks(range(len(comp_names)))
            axes[3].set_xticklabels(comp_names, rotation=45)
            axes[3].set_yticklabels(comp_names)
            axes[3].set_title("Reward Components Correlation")
            
            # Add correlation values
            for i in range(len(comp_names)):
                for j in range(len(comp_names)):
                    axes[3].text(j, i, f'{corr_matrix[i, j]:.2f}', 
                               ha='center', va='center', color='black')
            
            plt.colorbar(im, ax=axes[3])
        
        # 5. Episode success rate
        success_rates = [ep['success'] for ep in episodes]
        success_count = np.sum(success_rates)
        total_count = len(success_rates)
        
        axes[4].pie([success_count, total_count - success_count], 
                   labels=['Success', 'Failure'], 
                   colors=['lightgreen', 'lightcoral'],
                   autopct='%1.1f%%')
        axes[4].set_title(f"Success Rate: {success_count}/{total_count}")
        
        # 6. Performance metrics
        metrics = {
            'Avg Distance': np.mean(episode_distances),
            'Avg Reward': np.mean(episode_rewards),
            'Success Rate': success_count / total_count,
            'Std Distance': np.std(episode_distances),
            'Std Reward': np.std(episode_rewards)
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[5].bar(metric_names, metric_values, color='lightsteelblue')
        axes[5].set_title("Performance Metrics")
        axes[5].set_ylabel("Value")
        axes[5].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[5].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle("Comprehensive Reward Analysis", fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = "reward_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_robustness_matrix(self, robustness_data: Dict[str, Dict[str, float]], 
                             save: bool = True) -> str:
        """Plot robustness matrix across physics conditions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        conditions = list(robustness_data.keys())
        metrics = ['success_rate', 'avg_distance', 'avg_reward', 'episode_count']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            values = [robustness_data[cond].get(metric, 0) for cond in conditions]
            
            bars = ax.bar(conditions, values, color=plt.cm.viridis(np.linspace(0, 1, len(conditions))))
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylabel("Value")
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                       f'{value:.2f}', ha='center', va='bottom')
        
        plt.suptitle("Robustness Across Physics Conditions", fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = "robustness_matrix.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def plot_foot_clearance_analysis(self, episodes: List[Dict], save: bool = True) -> str:
        """Plot detailed foot clearance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        all_left_foot = []
        all_right_foot = []
        all_clearance = []
        
        for ep in episodes:
            if 'trajectory_data' in ep and 'foot_positions' in ep['trajectory_data']:
                foot_positions = np.array(ep['trajectory_data']['foot_positions'])
                left_foot = foot_positions[:, 0]
                right_foot = foot_positions[:, 1]
                clearance = np.maximum(left_foot, right_foot)
                
                all_left_foot.extend(left_foot)
                all_right_foot.extend(right_foot)
                all_clearance.extend(clearance)
        
        if not all_clearance:
            # No data available
            for ax in axes.flatten():
                ax.text(0.5, 0.5, 'No foot clearance data available', 
                       ha='center', va='center', transform=ax.transAxes)
            return ""
        
        # 1. Foot clearance over time (sample episode)
        if episodes:
            sample_ep = episodes[0]
            if 'trajectory_data' in sample_ep and 'foot_positions' in sample_ep['trajectory_data']:
                foot_positions = np.array(sample_ep['trajectory_data']['foot_positions'])
                time_steps = np.arange(len(foot_positions)) * 0.01
                
                axes[0].plot(time_steps, foot_positions[:, 0], label='Left Foot', color='blue')
                axes[0].plot(time_steps, foot_positions[:, 1], label='Right Foot', color='red')
                axes[0].axhline(y=0.03, color='green', linestyle='--', alpha=0.7, label='Target Clearance')
                axes[0].set_title("Foot Clearance Over Time (Sample Episode)")
                axes[0].set_xlabel("Time (s)")
                axes[0].set_ylabel("Foot Height (m)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
        
        # 2. Clearance distribution
        axes[1].hist(all_clearance, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1].axvline(np.mean(all_clearance), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_clearance):.3f}')
        axes[1].axvline(0.03, color='green', linestyle='--', alpha=0.7, label='Target: 0.03m')
        axes[1].set_title("Foot Clearance Distribution")
        axes[1].set_xlabel("Clearance (m)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        
        # 3. Left vs Right foot scatter
        axes[2].scatter(all_left_foot, all_right_foot, alpha=0.5, color='purple')
        axes[2].plot([0, max(max(all_left_foot), max(all_right_foot))], 
                    [0, max(max(all_left_foot), max(all_right_foot))], 
                    'r--', alpha=0.7, label='Perfect Symmetry')
        axes[2].set_title("Left vs Right Foot Height")
        axes[2].set_xlabel("Left Foot Height (m)")
        axes[2].set_ylabel("Right Foot Height (m)")
        axes[2].legend()
        
        # 4. Clearance statistics by episode
        episode_clearances = []
        episode_ids = []
        
        for i, ep in enumerate(episodes):
            if 'trajectory_data' in ep and 'foot_positions' in ep['trajectory_data']:
                foot_positions = np.array(ep['trajectory_data']['foot_positions'])
                clearance = np.maximum(foot_positions[:, 0], foot_positions[:, 1])
                episode_clearances.append(np.mean(clearance))
                episode_ids.append(i)
        
        if episode_clearances:
            bars = axes[3].bar(episode_ids, episode_clearances, color='lightcoral')
            axes[3].axhline(y=0.03, color='green', linestyle='--', alpha=0.7, label='Target Clearance')
            axes[3].set_title("Average Clearance by Episode")
            axes[3].set_xlabel("Episode ID")
            axes[3].set_ylabel("Avg Clearance (m)")
            axes[3].legend()
            
            # Add value labels
            for bar, value in zip(bars, episode_clearances):
                axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle("Foot Clearance Analysis", fontsize=16)
        plt.tight_layout()
        
        if save:
            filename = "foot_clearance_analysis.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath
        else:
            plt.show()
            return ""
    
    def _plot_joint_trajectories(self, ax, data: Dict, time: np.ndarray, 
                               title: str, color: str = 'blue'):
        """Plot joint angle trajectories."""
        if 'joint_positions' not in data:
            ax.text(0.5, 0.5, 'No joint data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        joint_positions = np.array(data['joint_positions'])
        
        ax.plot(time, joint_positions[:, 0], label='Hip', color=color, alpha=0.8)
        ax.plot(time, joint_positions[:, 1], label='Left Knee', color=color, alpha=0.6)
        ax.plot(time, joint_positions[:, 2], label='Right Knee', color=color, alpha=0.6)
        
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_control_signals(self, ax, data: Dict, time: np.ndarray, 
                            title: str, color: str = 'blue'):
        """Plot control signals."""
        if 'actions' not in data:
            ax.text(0.5, 0.5, 'No action data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        actions = np.array(data['actions'])
        
        ax.plot(time, actions[:, 0], label='Hip Action', color=color, alpha=0.8)
        ax.plot(time, actions[:, 1], label='Left Knee Action', color=color, alpha=0.6)
        ax.plot(time, actions[:, 2], label='Right Knee Action', color=color, alpha=0.6)
        
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_reward_components(self, ax, data: Dict, time: np.ndarray):
        """Plot reward components over time."""
        if 'reward_components' not in data:
            ax.text(0.5, 0.5, 'No reward data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        reward_components = data['reward_components']
        
        # Extract component names and values
        component_names = []
        component_values = []
        
        for comp_data in reward_components:
            if isinstance(comp_data, dict):
                for key, value in comp_data.items():
                    if key.startswith('r_') and isinstance(value, (int, float)):
                        if key not in component_names:
                            component_names.append(key)
                            component_values.append([])
                        component_values[component_names.index(key)].append(value)
        
        # Plot components
        colors = plt.cm.tab10(np.linspace(0, 1, len(component_names)))
        for i, (name, values) in enumerate(zip(component_names, component_values)):
            if len(values) == len(time):
                ax.plot(time, values, label=name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_portrait(self, ax, data: Dict, title: str, color: str = 'blue'):
        """Plot phase portrait (position vs velocity)."""
        if 'joint_positions' not in data or 'joint_velocities' not in data:
            ax.text(0.5, 0.5, 'No phase data available', ha='center', va='center', transform=ax.transAxes)
            return
        
        joint_positions = np.array(data['joint_positions'])
        joint_velocities = np.array(data['joint_velocities'])
        
        # Plot hip phase portrait
        ax.plot(joint_positions[:, 0], joint_velocities[:, 0], color=color, alpha=0.7)
        ax.scatter(joint_positions[0, 0], joint_velocities[0, 0], color='green', s=100, label='Start')
        ax.scatter(joint_positions[-1, 0], joint_velocities[-1, 0], color='red', s=100, label='End')
        
        ax.set_xlabel("Hip Position (rad)")
        ax.set_ylabel("Hip Velocity (rad/s)")
        ax.legend()
        ax.grid(True, alpha=0.3)


def create_comprehensive_plots(evaluation_results, output_dir: str = "plots") -> List[str]:
    """Create all comprehensive plots from evaluation results."""
    visualizer = TrajectoryVisualizer(output_dir)
    plot_files = []
    
    # 1. Episode comparison plots
    for i, episode in enumerate(evaluation_results.episodes[:3]):  # First 3 episodes
        if 'trajectory_data' in episode:
            # Create dummy FSM data for comparison (in practice, you'd load real FSM data)
            fsm_data = {
                'observations': episode['trajectory_data']['observations'],
                'actions': episode['trajectory_data']['fsm_actions'],
                'joint_positions': episode['trajectory_data']['joint_positions'],
                'joint_velocities': episode['trajectory_data']['joint_velocities'],
                'reward_components': []
            }
            
            plot_file = visualizer.plot_episode_comparison(
                episode['trajectory_data'], fsm_data, i, save=True
            )
            plot_files.append(plot_file)
    
    # 2. Reward analysis
    episode_data = []
    for episode in evaluation_results.episodes:
        episode_data.append({
            'total_reward': episode.total_reward,
            'distance': episode.distance,
            'success': episode.success,
            'reward_components': episode.reward_components,
            'trajectory_data': episode.trajectory_data
        })
    
    plot_file = visualizer.plot_reward_analysis(episode_data, save=True)
    plot_files.append(plot_file)
    
    # 3. Robustness matrix
    plot_file = visualizer.plot_robustness_matrix(evaluation_results.robustness_matrix, save=True)
    plot_files.append(plot_file)
    
    # 4. Foot clearance analysis
    plot_file = visualizer.plot_foot_clearance_analysis(episode_data, save=True)
    plot_files.append(plot_file)
    
    return plot_files


if __name__ == "__main__":
    # Example usage
    visualizer = TrajectoryVisualizer("example_plots")
    
    # Create sample data
    sample_data = {
        'observations': [np.random.randn(11) for _ in range(100)],
        'actions': [np.random.randn(3) for _ in range(100)],
        'joint_positions': [np.random.randn(3) for _ in range(100)],
        'joint_velocities': [np.random.randn(3) for _ in range(100)],
        'reward_components': [{'r_dx': 0.1, 'r_pitch': -0.05} for _ in range(100)]
    }
    
    # Generate plots
    visualizer.plot_episode_comparison(sample_data, sample_data, 0, save=True)
    print("Visualization complete!")
