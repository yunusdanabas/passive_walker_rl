#!/usr/bin/env python3
"""
Simple FSM Data Visualizer

Creates visualization plots from collected FSM data.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from passive_walker.config.paths import PLOTS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_fsm_data(data_dir: str, output_file: str = None):
    """Visualize FSM collected data."""
    data_path = Path(data_dir)
    
    # Find all episode files
    episode_files = sorted(data_path.glob('episode_*.npz'))
    
    if not episode_files:
        print(f"No episode files found in {data_dir}")
        return
    
    print(f"Found {len(episode_files)} episode files")
    
    # Load first 10 episodes for visualization
    episodes = []
    for f in episode_files[:10]:
        data = np.load(f)
        episodes.append(data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FSM Data Collection Summary', fontsize=16, fontweight='bold')
    
    # 1. Hip angles over time
    ax = axes[0, 0]
    for i, ep in enumerate(episodes[:5]):
        time = np.arange(len(ep['obs'])) * 0.01
        ax.plot(time, ep['obs'][:, 5], alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Hip Angle (rad)', fontsize=12)
    ax.set_title('Hip Angles Over Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Desired joint positions
    ax = axes[0, 1]
    if 'info_qdes' in episodes[0]:
        for j, joint_name in enumerate(['Hip', 'Left Knee', 'Right Knee']):
            for i, ep in enumerate(episodes[:3]):
                time = np.arange(len(ep['info_qdes'])) * 0.01
                ax.plot(time, ep['info_qdes'][:, j], alpha=0.5, label=joint_name if i == 0 else '')
    else:
        ax.text(0.5, 0.5, 'No qdes data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Desired Position (rad)', fontsize=12)
    ax.set_title('Desired Joint Positions', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. FSM states
    ax = axes[1, 0]
    for i, ep in enumerate(episodes[:5]):
        time = np.arange(len(ep['info_fsm_hip'])) * 0.01
        ax.plot(time, ep['info_fsm_hip'], alpha=0.6, label=f'Episode {i+1}')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('FSM Hip State', fontsize=12)
    ax.set_title('FSM Hip State Transitions', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Reward distribution
    ax = axes[1, 1]
    episode_returns = []
    for ep in episodes:
        episode_returns.append(np.sum(ep['rew']))
    ax.hist(episode_returns, bins=20, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Episode Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Episode Return Distribution (mean={np.mean(episode_returns):.1f})', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_file:
        out_path = Path(redirect_legacy_dir(output_file))
        ensure_dir_exists(out_path.parent)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {out_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize FSM collected data")
    parser.add_argument("data_dir", type=str, help="Directory containing FSM episode files")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    visualize_fsm_data(args.data_dir, args.output)


if __name__ == "__main__":
    main()

