"""
Behavioral Analysis Module

Analyzes NN vs FSM control patterns, trajectories, and decision-making behavior.
Generates comprehensive visualizations with minimal text output.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.models.models_torch import TorchMLP, TorchMLPLarge
from passive_walker.bc.dataset import Normalizer
from passive_walker.bc.play import _assemble_action_torch


def load_model_and_normalizer(checkpoint_path: str, meta_path: str):
    """Load trained model and normalizer."""
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    architecture = meta.get('architecture', 'TorchMLPLarge')
    
    # Load model
    if architecture == 'TorchMLPLarge':
        model = TorchMLPLarge(meta['input_dim'], meta['output_dim'])
    else:
        model = TorchMLP(meta['input_dim'], meta['output_dim'])
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    # Load normalizer
    normalizer = Normalizer(meta['input_dim'])
    if 'normalizer_mean' in meta and 'normalizer_std' in meta:
        normalizer.mean = np.array(meta['normalizer_mean'])
        normalizer.std = np.array(meta['normalizer_std'])
    elif 'normalizer_stats' in meta:
        normalizer.mean = np.array(meta['normalizer_stats']['mean'])
        normalizer.std = np.array(meta['normalizer_stats']['std'])
    
    return model, normalizer, meta


def collect_episode_data(env, model, normalizer, episodes: int = 10, model_type: str = "nn", metadata: dict = None):
    """Collect episode data for analysis with frame stacking support."""
    data = {
        'times': [],
        'actions': [],
        'rewards': [],
        'positions': [],
        'velocities': [],
        'joint_angles': [],
        'joint_velocities': []
    }
    
    # Check if model uses frame stacking
    frame_stack = 1
    expected_input_dim = normalizer.mean.shape[0] if hasattr(normalizer, 'mean') else 11
    obs_dim = 11
    if expected_input_dim > obs_dim:
        frame_stack = expected_input_dim // obs_dim
    
    for ep in range(episodes):
        obs, _ = env.reset()
        obs_buffer = []
        
        while env.data.time < 20.0:
            if model_type == "nn" and model is not None:
                # Handle frame stacking
                if frame_stack > 1:
                    obs_buffer.append(obs)
                    if len(obs_buffer) > frame_stack:
                        obs_buffer.pop(0)
                    while len(obs_buffer) < frame_stack:
                        obs_buffer.append(obs)
                    x = np.concatenate(obs_buffer).astype(np.float32)
                else:
                    x = obs.astype(np.float32)
                
                x_normalized = normalizer.apply(x[None, :]).astype(np.float32)
                with torch.no_grad():
                    model_output = model(torch.tensor(x_normalized, dtype=torch.float32))[0].numpy()
                
                mode = env.mode
                section = metadata.get('section', 'both')  # Use section from metadata
                action = _assemble_action_torch(section, model_output, None, None, None, "act")
            else:
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            # Collect data
            data['times'].append(env.data.time)
            data['actions'].append(action.copy())
            data['rewards'].append(0.0)
            data['positions'].append([env.data.qpos[env.qpos_x], env.data.qpos[env.qpos_z]])
            data['velocities'].append([env.data.qvel[env.dof_x], env.data.qvel[env.dof_z]])
            data['joint_angles'].append([
                env.data.qpos[env.qpos_hip],
                env.data.qpos[env.qpos_lk],
                env.data.qpos[env.qpos_rk]
            ])
            data['joint_velocities'].append([
                env.data.qvel[env.qvel_hip],
                env.data.qvel[env.qvel_lk],
                env.data.qvel[env.qvel_rk]
            ])
            
            obs, reward, done, info = env.step(action)
            data['rewards'][-1] = reward
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def create_control_pattern_plots(nn_data: Dict, fsm_data: Dict, output_dir: Path):
    """Create comprehensive control pattern visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Action trajectories over time
    for i, joint in enumerate(['Hip', 'Left Knee', 'Right Knee']):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(nn_data['times'][:500], nn_data['actions'][:500, i], 
                alpha=0.7, label='NN', linewidth=1.5)
        ax.plot(fsm_data['times'][:500], fsm_data['actions'][:500, i], 
                alpha=0.7, label='FSM', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Action', fontsize=10)
        ax.set_title(f'{joint} Control', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Action distributions
    for i, joint in enumerate(['Hip', 'Left Knee', 'Right Knee']):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(nn_data['actions'][:, i], bins=50, alpha=0.6, label='NN', density=True)
        ax.hist(fsm_data['actions'][:, i], bins=50, alpha=0.6, label='FSM', density=True)
        ax.set_xlabel('Action Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{joint} Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Joint angle phase plots
    for i, joint in enumerate(['Hip', 'Left Knee', 'Right Knee']):
        ax = fig.add_subplot(gs[2, i])
        ax.scatter(nn_data['joint_angles'][:1000, i], nn_data['joint_velocities'][:1000, i],
                  s=1, alpha=0.3, label='NN', c='blue')
        ax.scatter(fsm_data['joint_angles'][:1000, i], fsm_data['joint_velocities'][:1000, i],
                  s=1, alpha=0.3, label='FSM', c='orange')
        ax.set_xlabel('Angle (rad)', fontsize=10)
        ax.set_ylabel('Velocity (rad/s)', fontsize=10)
        ax.set_title(f'{joint} Phase Portrait', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Behavioral Analysis: NN vs FSM Control Patterns', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'control_patterns.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_trajectory_comparison(nn_data: Dict, fsm_data: Dict, output_dir: Path):
    """Create trajectory comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Forward position over time
    ax = axes[0, 0]
    ax.plot(nn_data['times'][:1000], nn_data['positions'][:1000, 0], 
            label='NN', linewidth=2, alpha=0.8)
    ax.plot(fsm_data['times'][:1000], fsm_data['positions'][:1000, 0], 
            label='FSM', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('X Position (m)', fontsize=12)
    ax.set_title('Forward Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Reward over time
    ax = axes[0, 1]
    nn_reward_smooth = np.convolve(nn_data['rewards'], np.ones(50)/50, mode='valid')
    fsm_reward_smooth = np.convolve(fsm_data['rewards'], np.ones(50)/50, mode='valid')
    ax.plot(nn_data['times'][:len(nn_reward_smooth)], nn_reward_smooth, 
            label='NN', linewidth=2, alpha=0.8)
    ax.plot(fsm_data['times'][:len(fsm_reward_smooth)], fsm_reward_smooth, 
            label='FSM', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Reward (smoothed)', fontsize=12)
    ax.set_title('Performance Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Velocity comparison
    ax = axes[1, 0]
    nn_vel = np.linalg.norm(nn_data['velocities'][:1000], axis=1)
    fsm_vel = np.linalg.norm(fsm_data['velocities'][:1000], axis=1)
    ax.plot(nn_data['times'][:1000], nn_vel, label='NN', linewidth=2, alpha=0.8)
    ax.plot(fsm_data['times'][:1000], fsm_vel, label='FSM', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (m/s)', fontsize=12)
    ax.set_title('Walking Speed', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Performance metrics table
    ax = axes[1, 1]
    ax.axis('off')
    
    metrics = [
        ['Metric', 'NN', 'FSM', 'Difference'],
        ['Avg Reward', f'{nn_data["rewards"].mean():.3f}', 
         f'{fsm_data["rewards"].mean():.3f}', 
         f'{(nn_data["rewards"].mean() - fsm_data["rewards"].mean()):.3f}'],
        ['Total Distance', f'{nn_data["positions"][-1, 0]:.2f}m',
         f'{fsm_data["positions"][-1, 0]:.2f}m',
         f'{(nn_data["positions"][-1, 0] - fsm_data["positions"][-1, 0]):.2f}m'],
        ['Avg Speed', f'{nn_vel.mean():.3f}m/s',
         f'{fsm_vel.mean():.3f}m/s',
         f'{(nn_vel.mean() - fsm_vel.mean()):.3f}m/s'],
        ['Action Std (Hip)', f'{nn_data["actions"][:, 0].std():.3f}',
         f'{fsm_data["actions"][:, 0].std():.3f}',
         f'{(nn_data["actions"][:, 0].std() - fsm_data["actions"][:, 0].std()):.3f}']
    ]
    
    table = ax.table(cellText=metrics, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Behavioral Analysis: Trajectory Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'trajectory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def run_behavioral_analysis(checkpoint_path: str, meta_path: str, 
                           output_dir: Path, episodes: int = 10) -> Dict:
    """
    Main behavioral analysis runner.
    
    Returns:
        Dictionary with paths to generated figures and minimal metrics.
    """
    print("📊 Running Behavioral Analysis...")
    print(f"   Model: {checkpoint_path}")
    print(f"   Episodes: {episodes}")
    
    # Load model
    model, normalizer, meta = load_model_and_normalizer(checkpoint_path, meta_path)
    
    # Create environments
    env_nn = PassiveWalkerEnv(mode="research", use_gui=False)
    env_fsm = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    # Collect data
    print("   Collecting NN data...")
    nn_data = collect_episode_data(env_nn, model, normalizer, episodes, "nn", meta)
    
    print("   Collecting FSM data...")
    fsm_data = collect_episode_data(env_fsm, None, None, episodes, "fsm")
    
    # Generate visualizations
    print("   Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig1 = create_control_pattern_plots(nn_data, fsm_data, output_dir)
    fig2 = create_trajectory_comparison(nn_data, fsm_data, output_dir)
    
    # Compute summary metrics
    metrics = {
        'nn_avg_reward': float(nn_data['rewards'].mean()),
        'fsm_avg_reward': float(fsm_data['rewards'].mean()),
        'reward_improvement': float(nn_data['rewards'].mean() - fsm_data['rewards'].mean()),
        'nn_total_distance': float(nn_data['positions'][-1, 0]),
        'fsm_total_distance': float(fsm_data['positions'][-1, 0])
    }
    
    env_nn.close()
    env_fsm.close()
    
    print(f"   ✅ Analysis complete!")
    print(f"   📊 Reward improvement: {metrics['reward_improvement']:.3f}")
    
    return {
        'figures': [str(fig1), str(fig2)],
        'metrics': metrics
    }


if __name__ == "__main__":
    # Example usage
    checkpoint = "checkpoints/torch_both_seed123_ep1_steps180000.pt"
    meta = "checkpoints/torch_both_seed123_ep1_steps180000_meta.json"
    output = Path("results/latest_analysis/figures")
    
    results = run_behavioral_analysis(checkpoint, meta, output, episodes=5)
    print(f"\n📁 Figures saved to: {results['figures']}")

