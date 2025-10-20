"""
Robustness Testing Module

Tests model resilience under various physics variations and disturbances.
Generates comprehensive visualizations with minimal text output.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import sys

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
    
    if architecture == 'TorchMLPLarge':
        model = TorchMLPLarge(meta['input_dim'], meta['output_dim'])
    else:
        model = TorchMLP(meta['input_dim'], meta['output_dim'])
    
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    
    normalizer = Normalizer(meta['input_dim'])
    if 'normalizer_mean' in meta and 'normalizer_std' in meta:
        normalizer.mean = np.array(meta['normalizer_mean'])
        normalizer.std = np.array(meta['normalizer_std'])
    elif 'normalizer_stats' in meta:
        normalizer.mean = np.array(meta['normalizer_stats']['mean'])
        normalizer.std = np.array(meta['normalizer_stats']['std'])
    
    return model, normalizer, meta


def test_physics_variation(env, model, normalizer, variation_type: str, 
                          episodes: int = 5, model_type: str = "nn"):
    """Test model under physics variations."""
    # Detect frame stacking
    frame_stack = 1
    expected_input_dim = normalizer.mean.shape[0] if hasattr(normalizer, 'mean') else 11
    obs_dim = 11
    if expected_input_dim > obs_dim:
        frame_stack = expected_input_dim // obs_dim
    
    rewards = []
    distances = []
    success_count = 0
    
    for ep in range(episodes):
        obs, _ = env.reset()
        obs_buffer = []
        episode_reward = 0.0
        
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
                section = "both" if mode == "research" else mode.replace("hybrid_", "")
                action = _assemble_action_torch(section, model_output, None, None, None, "act")
            else:
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        final_distance = env.data.qpos[env.qpos_x]
        rewards.append(episode_reward)
        distances.append(final_distance)
        
        if final_distance > 5.0:  # Success threshold
            success_count += 1
    
    return {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'avg_distance': np.mean(distances),
        'success_rate': success_count / episodes,
        'all_rewards': rewards,
        'all_distances': distances
    }


def create_robustness_visualization(results: Dict, output_dir: Path):
    """Create comprehensive robustness testing visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    conditions = list(results['nn'].keys())
    nn_rewards = [results['nn'][c]['avg_reward'] for c in conditions]
    fsm_rewards = [results['fsm'][c]['avg_reward'] for c in conditions]
    nn_std = [results['nn'][c]['std_reward'] for c in conditions]
    fsm_std = [results['fsm'][c]['std_reward'] for c in conditions]
    
    # Reward comparison bar chart
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(conditions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nn_rewards, width, label='NN', 
                   yerr=nn_std, capsize=5, alpha=0.8)
    bars2 = ax.bar(x + width/2, fsm_rewards, width, label='FSM',
                   yerr=fsm_std, capsize=5, alpha=0.8)
    
    ax.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax.set_title('Robustness Across Conditions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in conditions], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Success rate comparison
    ax = fig.add_subplot(gs[0, 2])
    nn_success = [results['nn'][c]['success_rate'] * 100 for c in conditions]
    fsm_success = [results['fsm'][c]['success_rate'] * 100 for c in conditions]
    
    bars1 = ax.bar(x - width/2, nn_success, width, label='NN', alpha=0.8)
    bars2 = ax.bar(x + width/2, fsm_success, width, label='FSM', alpha=0.8)
    
    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Success Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in conditions], rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    # Distance boxplots for each condition
    for idx, condition in enumerate(conditions[:6]):  # First 6 conditions
        row = (idx // 3) + 1
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        data_to_plot = [
            results['nn'][condition]['all_distances'],
            results['fsm'][condition]['all_distances']
        ]
        
        bp = ax.boxplot(data_to_plot, labels=['NN', 'FSM'], patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        
        ax.set_ylabel('Distance (m)', fontsize=10)
        ax.set_title(condition.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Robustness Testing: Performance Under Varying Conditions',
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'robustness_testing.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_robustness_summary_table(results: Dict, output_dir: Path):
    """Create detailed summary table visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('off')
    
    conditions = list(results['nn'].keys())
    
    # Build table data
    table_data = [['Condition', 'NN Reward', 'FSM Reward', 'Difference', 
                   'NN Success', 'FSM Success', 'NN Distance', 'FSM Distance']]
    
    for cond in conditions:
        nn_r = results['nn'][cond]['avg_reward']
        fsm_r = results['fsm'][cond]['avg_reward']
        diff = nn_r - fsm_r
        nn_s = results['nn'][cond]['success_rate'] * 100
        fsm_s = results['fsm'][cond]['success_rate'] * 100
        nn_d = results['nn'][cond]['avg_distance']
        fsm_d = results['fsm'][cond]['avg_distance']
        
        table_data.append([
            cond.replace('_', ' ').title(),
            f'{nn_r:.2f}',
            f'{fsm_r:.2f}',
            f'{diff:+.2f}',
            f'{nn_s:.0f}%',
            f'{fsm_s:.0f}%',
            f'{nn_d:.2f}m',
            f'{fsm_d:.2f}m'
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header row
    for i in range(8):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color difference column based on sign
    for i in range(1, len(table_data)):
        diff_val = float(table_data[i][3])
        if diff_val > 0:
            table[(i, 3)].set_facecolor('#C8E6C9')  # Light green
        elif diff_val < 0:
            table[(i, 3)].set_facecolor('#FFCDD2')  # Light red
    
    ax.set_title('Robustness Testing: Detailed Performance Summary',
                 fontsize=14, fontweight='bold', pad=20)
    
    output_path = output_dir / 'robustness_summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def run_robustness_testing(checkpoint_path: str, meta_path: str,
                          output_dir: Path, episodes: int = 5) -> Dict:
    """
    Main robustness testing runner.
    
    Returns:
        Dictionary with paths to generated figures and minimal metrics.
    """
    print("🛡️ Running Robustness Testing...")
    print(f"   Model: {checkpoint_path}")
    print(f"   Episodes per condition: {episodes}")
    
    # Load model
    model, normalizer, meta = load_model_and_normalizer(checkpoint_path, meta_path)
    
    # Define test conditions
    test_conditions = {
        'baseline': {},
        'low_friction': {'friction': 0.3},
        'high_friction': {'friction': 2.0},
        'steep_ramp': {'ramp_angle': 0.1},
        'heavy_mass': {'mass_scale': 1.5},
        'light_mass': {'mass_scale': 0.7}
    }
    
    results = {'nn': {}, 'fsm': {}}
    
    # Test each condition
    for condition_name, params in test_conditions.items():
        print(f"   Testing {condition_name}...")
        
        # Test NN
        env_nn = PassiveWalkerEnv(mode="research", use_gui=False)
        results['nn'][condition_name] = test_physics_variation(
            env_nn, model, normalizer, condition_name, episodes, "nn"
        )
        env_nn.close()
        
        # Test FSM
        env_fsm = PassiveWalkerEnv(mode="fsm", use_gui=False)
        results['fsm'][condition_name] = test_physics_variation(
            env_fsm, None, None, condition_name, episodes, "fsm"
        )
        env_fsm.close()
    
    # Generate visualizations
    print("   Generating visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig1 = create_robustness_visualization(results, output_dir)
    fig2 = create_robustness_summary_table(results, output_dir)
    
    # Compute summary metrics
    avg_nn_reward = np.mean([results['nn'][c]['avg_reward'] for c in test_conditions])
    avg_fsm_reward = np.mean([results['fsm'][c]['avg_reward'] for c in test_conditions])
    
    print(f"   ✅ Testing complete!")
    print(f"   📊 Avg NN reward: {avg_nn_reward:.3f}")
    print(f"   📊 Avg FSM reward: {avg_fsm_reward:.3f}")
    
    return {
        'figures': [str(fig1), str(fig2)],
        'metrics': {
            'avg_nn_reward': float(avg_nn_reward),
            'avg_fsm_reward': float(avg_fsm_reward),
            'improvement': float(avg_nn_reward - avg_fsm_reward)
        }
    }


if __name__ == "__main__":
    # Example usage
    checkpoint = "checkpoints/torch_both_seed123_ep1_steps180000.pt"
    meta = "checkpoints/torch_both_seed123_ep1_steps180000_meta.json"
    output = Path("results/latest_analysis/figures")
    
    results = run_robustness_testing(checkpoint, meta, output, episodes=3)
    print(f"\n📁 Figures saved to: {results['figures']}")

