#!/usr/bin/env python3
"""
Model Performance Plotting and Analysis

This script collects data during model evaluation runs and creates comprehensive plots
comparing different models (hip, knees, both) across various metrics.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from passive_walker.bc.play import play_torch, play_jax
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.utils import load_checkpoint, Normalizer
from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge


class ModelDataCollector:
    """Collects detailed data during model evaluation for plotting."""
    
    def __init__(self):
        self.episode_data = []
        self.current_episode = []
        
    def reset_episode(self):
        """Start collecting data for a new episode."""
        self.current_episode = []
        
    def add_step(self, obs, action, reward, done, info, model_output=None, fsm_output=None):
        """Add step data to current episode."""
        step_data = {
            'time': info.get('time', 0.0),
            'reward': reward,
            'done': done,
            'dx': info.get('dx', 0.0),
            'pitch_abs': info.get('pitch_abs', 0.0),
            'torso_z': info.get('torso_z', 0.0),
            'u_abs_sum': info.get('u_abs_sum', 0.0),
            'control_effort': info.get('u', np.array([0, 0, 0])),
            'desired_positions': info.get('qdes', np.array([0, 0, 0])),
            'fsm_states': {
                'hip': info.get('fsm_hip', 0),
                'knee1': info.get('fsm_k1', 0),
                'knee2': info.get('fsm_k2', 0)
            },
            'fell': info.get('fell', False),
            'obs': obs.copy(),
            'action': action.copy(),
            'model_output': model_output.copy() if model_output is not None else None,
            'fsm_output': fsm_output.copy() if fsm_output is not None else None
        }
        self.current_episode.append(step_data)
        
    def finish_episode(self):
        """Finish current episode and add to collected data."""
        if self.current_episode:
            self.episode_data.append(self.current_episode.copy())
            
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics across all episodes."""
        if not self.episode_data:
            return {}
            
        total_steps = sum(len(ep) for ep in self.episode_data)
        total_time = sum(ep[-1]['time'] if ep else 0 for ep in self.episode_data)
        total_reward = sum(sum(step['reward'] for step in ep) for ep in self.episode_data)
        success_rate = sum(1 for ep in self.episode_data if not ep[-1]['fell']) / len(self.episode_data)
        
        # Average distance (assuming 100Hz control)
        avg_distance = sum(sum(step['dx'] for step in ep) for ep in self.episode_data) / len(self.episode_data)
        
        # Stability metrics
        all_pitch = [step['pitch_abs'] for ep in self.episode_data for step in ep]
        avg_pitch = np.mean(all_pitch) if all_pitch else 0
        
        return {
            'episodes': len(self.episode_data),
            'total_steps': total_steps,
            'total_time': total_time,
            'avg_episode_time': total_time / len(self.episode_data),
            'total_reward': total_reward,
            'avg_reward': total_reward / len(self.episode_data),
            'success_rate': success_rate,
            'avg_distance': avg_distance,
            'avg_pitch': avg_pitch,
            'avg_steps_per_episode': total_steps / len(self.episode_data)
        }


def collect_fsm_data(episodes: int = 5, seconds: float = 20.0) -> ModelDataCollector:
    """Run FSM baseline and collect detailed data."""
    
    # Create FSM-only environment
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    # Data collector
    collector = ModelDataCollector()
    
    print(f"Collecting FSM baseline data...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        collector.reset_episode()
        episode_steps = 0
        
        while env.data.time < seconds and episode_steps < seconds * 100:  # 100Hz
            # FSM mode - actions are ignored, FSM controls everything
            action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # FSM ignores this
            
            # Get FSM desired outputs
            env.fsm.update(env.data, env.model)
            fsm_hip = env.fsm.desired_hip()
            fsm_lk, fsm_rk = env.fsm.desired_knees()
            fsm_output = np.array([fsm_hip, fsm_lk, fsm_rk])
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Collect data (no model output for FSM)
            collector.add_step(obs, action, reward, done, info, None, fsm_output)
            
            episode_steps += 1
            if done:
                break
                
        collector.finish_episode()
        print(f"  FSM Episode {ep+1}: {episode_steps} steps, {info.get('time', 0):.2f}s")
        
    env.close()
    return collector


def collect_model_data(ckpt_path: str, meta_path: str, episodes: int = 5, seconds: float = 20.0) -> ModelDataCollector:
    """Run model evaluation and collect detailed data."""
    import torch
    
    # Load metadata and model
    with open(meta_path, "r") as f:
        meta = json.load(f)
        
    model = TorchMLPLarge(in_dim=meta["input_dim"], out_dim=meta["output_dim"], hidden=512, dropout=0.1)
    model = load_checkpoint(ckpt_path, model)
    model.eval()
    
    # Setup normalizer
    normalizer = Normalizer(
        mean=np.array(meta["normalizer_mean"], dtype=np.float32),
        std=np.array(meta["normalizer_std"], dtype=np.float32)
    )
    
    # Create environment
    if meta["section"] == "hip":
        mode = "hybrid_hip"
    elif meta["section"] == "knees":
        mode = "hybrid_knees"
    elif meta["section"] in ("both", "both-adv"):
        mode = "research"
        
    env = PassiveWalkerEnv(mode=mode, use_gui=False)
    
    # Data collector
    collector = ModelDataCollector()
    
    print(f"Collecting data for {meta['section']} model...")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        collector.reset_episode()
        episode_steps = 0
        
        while env.data.time < seconds and episode_steps < seconds * 100:  # 100Hz
            # Get model prediction
            x = normalizer.apply(obs[None, :]).astype(np.float32)
            with torch.no_grad():
                model_output = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
                
            # Assemble action (simplified version of _assemble_action_torch)
            from passive_walker.bc.play import _assemble_action_torch
            action = _assemble_action_torch(meta["section"], model_output, None, None, None, meta.get("label_type", "act"))
            
            # Get FSM desired outputs for comparison
            env.fsm.update(env.data, env.model)
            fsm_hip = env.fsm.desired_hip()
            fsm_lk, fsm_rk = env.fsm.desired_knees()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Collect data
            collector.add_step(obs, action, reward, done, info, model_output, 
                             np.array([fsm_hip, fsm_lk, fsm_rk]))
            
            episode_steps += 1
            if done:
                break
                
        collector.finish_episode()
        print(f"  Episode {ep+1}: {episode_steps} steps, {info.get('time', 0):.2f}s")
        
    env.close()
    return collector


def create_time_series_plots(collected_data: Dict[str, ModelDataCollector], output_dir: str):
    """Create time series plots for different models."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Time Series Comparison', fontsize=16)
    
    # Plot for each episode, showing first successful episode if available
    colors = {'hip': 'blue', 'knees': 'green', 'both': 'red', 'fsm': 'orange'}
    
    for model_name, collector in collected_data.items():
        if not collector.episode_data:
            continue
            
        # Find best episode (longest without falling)
        best_episode = None
        for ep in collector.episode_data:
            if not ep[-1]['fell']:
                if best_episode is None or len(ep) > len(best_episode):
                    best_episode = ep
                    
        if best_episode is None:
            best_episode = collector.episode_data[0]  # Use first episode if all fell
            
        times = [step['time'] for step in best_episode]
        color = colors.get(model_name, 'black')
        
        # Plot 1: Horizontal velocity (dx)
        axes[0, 0].plot(times, [step['dx'] for step in best_episode], 
                       color=color, label=f'{model_name}', alpha=0.7)
        
        # Plot 2: Absolute pitch
        axes[0, 1].plot(times, [step['pitch_abs'] for step in best_episode], 
                       color=color, label=f'{model_name}', alpha=0.7)
        
        # Plot 3: Torso height
        axes[1, 0].plot(times, [step['torso_z'] for step in best_episode], 
                       color=color, label=f'{model_name}', alpha=0.7)
        
        # Plot 4: Control effort
        axes[1, 1].plot(times, [step['u_abs_sum'] for step in best_episode], 
                       color=color, label=f'{model_name}', alpha=0.7)
        
        # Plot 5: Cumulative reward
        cum_rewards = np.cumsum([step['reward'] for step in best_episode])
        axes[2, 0].plot(times, cum_rewards, color=color, label=f'{model_name}', alpha=0.7)
        
        # Plot 6: Desired joint positions (show hip as example)
        qdes_hip = [step['desired_positions'][0] for step in best_episode]
        axes[2, 1].plot(times, qdes_hip, color=color, label=f'{model_name} (hip)', alpha=0.7)
    
    # Set labels and legends
    axes[0, 0].set_title('Horizontal Velocity (m/s)')
    axes[0, 0].set_ylabel('dx (m/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Absolute Pitch Angle (rad)')
    axes[0, 1].set_ylabel('|pitch| (rad)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Torso Height (m)')
    axes[1, 0].set_ylabel('Z (m)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Control Effort')
    axes[1, 1].set_ylabel('|u| sum')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].set_title('Cumulative Reward')
    axes[2, 0].set_ylabel('Total Reward')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].set_title('Desired Hip Position (rad)')
    axes[2, 1].set_ylabel('qdes_hip (rad)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_series_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_comparison(collected_data: Dict[str, ModelDataCollector], output_dir: str):
    """Create bar charts comparing overall performance metrics."""
    
    # Calculate metrics for each model
    model_names = list(collected_data.keys())
    metrics = {
        'success_rate': [],
        'avg_reward': [],
        'avg_distance': [],
        'avg_episode_time': [],
        'avg_pitch': []
    }
    
    for model_name in model_names:
        stats = collected_data[model_name].get_summary_stats()
        for metric in metrics.keys():
            value = stats.get(metric, 0)
            metrics[metric].append(value)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16)
    
    # Define colors for each model type
    color_map = {'hip': 'blue', 'knees': 'green', 'both': 'red', 'fsm': 'orange'}
    colors = [color_map.get(name, 'gray') for name in model_names]
    
    # Success rate
    axes[0, 0].bar(model_names, metrics['success_rate'], color=colors)
    axes[0, 0].set_title('Success Rate')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(metrics['success_rate']):
        axes[0, 0].text(i, v + 0.01, f'{v:.2%}', ha='center')
    
    # Average reward
    axes[0, 1].bar(model_names, metrics['avg_reward'], color=colors)
    axes[0, 1].set_title('Average Episode Reward')
    axes[0, 1].set_ylabel('Reward')
    for i, v in enumerate(metrics['avg_reward']):
        axes[0, 1].text(i, v + max(metrics['avg_reward']) * 0.01, f'{v:.1f}', ha='center')
    
    # Average distance
    axes[0, 2].bar(model_names, metrics['avg_distance'], color=colors)
    axes[0, 2].set_title('Average Distance per Episode')
    axes[0, 2].set_ylabel('Distance (m)')
    for i, v in enumerate(metrics['avg_distance']):
        axes[0, 2].text(i, v + max(metrics['avg_distance']) * 0.01, f'{v:.2f}', ha='center')
    
    # Episode duration
    axes[1, 0].bar(model_names, metrics['avg_episode_time'], color=colors)
    axes[1, 0].set_title('Average Episode Duration')
    axes[1, 0].set_ylabel('Time (s)')
    for i, v in enumerate(metrics['avg_episode_time']):
        axes[1, 0].text(i, v + max(metrics['avg_episode_time']) * 0.01, f'{v:.1f}', ha='center')
    
    # Average pitch (stability)
    axes[1, 1].bar(model_names, metrics['avg_pitch'], color=colors)
    axes[1, 1].set_title('Average Pitch (Lower = Better Stability)')
    axes[1, 1].set_ylabel('|Pitch| (rad)')
    for i, v in enumerate(metrics['avg_pitch']):
        axes[1, 1].text(i, v + max(metrics['avg_pitch']) * 0.01, f'{v:.3f}', ha='center')
    
    # Steps per episode (remove subplot since we only have 5 metrics)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_joint_trajectory_plots(collected_data: Dict[str, ModelDataCollector], output_dir: str):
    """Create plots showing joint trajectories and control effort."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Joint Trajectories and Control Analysis', fontsize=16)
    
    colors = {'hip': 'blue', 'knees': 'green', 'both': 'red', 'fsm': 'orange'}
    joint_names = ['Hip', 'Left Knee', 'Right Knee']
    
    for model_name, collector in collected_data.items():
        if not collector.episode_data:
            continue
            
        # Get best episode
        best_episode = None
        for ep in collector.episode_data:
            if not ep[-1]['fell']:
                if best_episode is None or len(ep) > len(best_episode):
                    best_episode = ep
        if best_episode is None:
            best_episode = collector.episode_data[0]
            
        times = [step['time'] for step in best_episode]
        color = colors.get(model_name, 'black')
        
        # Plot desired positions for each joint
        for joint_idx in range(3):
            qdes_joint = [step['desired_positions'][joint_idx] for step in best_episode]
            axes[joint_idx, 0].plot(times, qdes_joint, color=color, 
                                  label=f'{model_name}', alpha=0.7, linewidth=1)
            
            # Plot control effort for each joint
            u_joint = [abs(step['control_effort'][joint_idx]) for step in best_episode]
            axes[joint_idx, 1].plot(times, u_joint, color=color, 
                                  label=f'{model_name}', alpha=0.7, linewidth=1)
    
    # Set labels
    for i in range(3):
        axes[i, 0].set_title(f'{joint_names[i]} - Desired Position')
        axes[i, 0].set_ylabel('Position')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].set_title(f'{joint_names[i]} - Control Effort')
        axes[i, 1].set_ylabel('|Control|')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'joint_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_phase_plot(collected_data: Dict[str, ModelDataCollector], output_dir: str):
    """Create phase plots (position vs velocity) for each joint."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Joint Phase Plots (Position vs Velocity)', fontsize=16)
    
    colors = {'hip': 'blue', 'knees': 'green', 'both': 'red', 'fsm': 'orange'}
    joint_names = ['Hip', 'Left Knee', 'Right Knee']
    
    for model_name, collector in collected_data.items():
        if not collector.episode_data:
            continue
            
        color = colors.get(model_name, 'black')
        alpha = 0.6
        
        # Get all episodes and concatenate
        for ep in collector.episode_data[:3]:  # Show first 3 episodes
            for step in ep:
                obs = step['obs']
                # Joint positions and velocities from observation (indices may vary)
                # Assuming standard observation format: [x, z, pitch, xd, zd, hip, lk, rk, hipd, lkd, rkd]
                jpos = obs[5:8]  # joint positions
                jvel = obs[8:11]  # joint velocities
                
                for joint_idx in range(3):
                    axes[joint_idx].scatter(jpos[joint_idx], jvel[joint_idx], 
                                          c=color, alpha=alpha, s=1)
    
    for i in range(3):
        axes[i].set_title(f'{joint_names[i]} Phase Plot')
        axes[i].set_xlabel('Position')
        axes[i].set_ylabel('Velocity')
        axes[i].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markersize=8, label=name) 
                         for name, color in colors.items()]
        axes[i].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_summary_report(collected_data: Dict[str, ModelDataCollector], output_dir: str):
    """Save a text summary report with all metrics."""
    
    report_path = os.path.join(output_dir, 'performance_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write("MODEL PERFORMANCE SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for model_name, collector in collected_data.items():
            stats = collector.get_summary_stats()
            
            f.write(f"{model_name.upper()} MODEL:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Episodes run: {stats.get('episodes', 0)}\n")
            f.write(f"Success rate: {stats.get('success_rate', 0):.2%}\n")
            f.write(f"Average reward: {stats.get('avg_reward', 0):.2f}\n")
            f.write(f"Average episode time: {stats.get('avg_episode_time', 0):.2f}s\n")
            f.write(f"Average distance: {stats.get('avg_distance', 0):.2f}m\n")
            f.write(f"Average pitch: {stats.get('avg_pitch', 0):.3f}rad\n")
            f.write(f"Average steps/episode: {stats.get('avg_steps_per_episode', 0):.0f}\n")
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Plot model performance data")
    parser.add_argument("--hip-ckpt", help="Path to hip model checkpoint")
    parser.add_argument("--hip-meta", help="Path to hip model metadata")
    parser.add_argument("--knees-ckpt", help="Path to knees model checkpoint") 
    parser.add_argument("--knees-meta", help="Path to knees model metadata")
    parser.add_argument("--both-ckpt", help="Path to both model checkpoint")
    parser.add_argument("--both-meta", help="Path to both model metadata")
    parser.add_argument("--output-dir", default="plots", help="Output directory for plots")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run per model")
    parser.add_argument("--seconds", type=float, default=20.0, help="Seconds per episode")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect data for each model
    collected_data = {}
    model_configs = [
        ("hip", args.hip_ckpt, args.hip_meta),
        ("knees", args.knees_ckpt, args.knees_meta),
        ("both", args.both_ckpt, args.both_meta)
    ]
    
    for model_name, ckpt_path, meta_path in model_configs:
        if ckpt_path and meta_path and os.path.exists(ckpt_path) and os.path.exists(meta_path):
            print(f"Collecting data for {model_name} model...")
            collected_data[model_name] = collect_model_data(ckpt_path, meta_path, args.episodes, args.seconds)
        else:
            print(f"Skipping {model_name} model - files not found")
    
    # Always collect FSM baseline data
    print("Collecting FSM baseline data...")
    collected_data['fsm'] = collect_fsm_data(args.episodes, args.seconds)
    
    if not collected_data:
        print("No valid model files found!")
        return
    
    # Generate all plots
    print("Generating plots...")
    create_time_series_plots(collected_data, args.output_dir)
    create_performance_comparison(collected_data, args.output_dir) 
    create_joint_trajectory_plots(collected_data, args.output_dir)
    create_phase_plot(collected_data, args.output_dir)
    save_summary_report(collected_data, args.output_dir)
    
    print(f"All plots saved to: {args.output_dir}/")
    print("Generated files:")
    print("  - time_series_comparison.png")
    print("  - performance_metrics_comparison.png") 
    print("  - joint_trajectories.png")
    print("  - phase_plots.png")
    print("  - performance_summary.txt")


if __name__ == "__main__":
    main()
