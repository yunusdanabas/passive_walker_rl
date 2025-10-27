#!/usr/bin/env python3
"""
Simple Visualization Tool

Plot training curves, trajectories, and evaluation results.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from passive_walker.config.paths import PPO_PLOTS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir


def plot_training_curves(log_dir: str, out_file: str = None):
    """Plot training curves from log directory."""
    try:
        # Try to load TensorBoard logs
        from torch.utils.tensorboard import SummaryReader
        
        reader = SummaryReader(log_dir)
        
        # Extract scalar data
        scalar_keys = reader.get_scalar_keys()
        
        if not scalar_keys:
            print("No scalar data found in logs")
            return
        
        fig, axes = plt.subplots(len(scalar_keys), 1, figsize=(10, 6*len(scalar_keys)))
        if len(scalar_keys) == 1:
            axes = [axes]
        
        for idx, key in enumerate(scalar_keys):
            data = reader.get_scalar(key)
            steps = [point.step for point in data]
            values = [point.value for point in data]
            
            axes[idx].plot(steps, values)
            axes[idx].set_title(key)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("Value")
            axes[idx].grid(True)
        
        plt.tight_layout()
        
        if out_file:
            plt.savefig(out_file)
            print(f"Plot saved to: {out_file}")
        else:
            plt.show()
            
        reader.close()
    except Exception as e:
        print(f"Error plotting training curves: {e}")


def plot_episode_trajectory(episode_data: dict, out_file: str = None):
    """Plot trajectory data from an episode."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        qpos = episode_data.get('qpos', [])
        qvel = episode_data.get('qvel', [])
        actions = episode_data.get('actions', [])
        rewards = episode_data.get('rewards', [])
        
        # Plot joint positions
        if qpos:
            axes[0, 0].plot(qpos)
            axes[0, 0].set_title("Joint Positions")
            axes[0, 0].set_xlabel("Step")
            axes[0, 0].set_ylabel("Position")
            axes[0, 0].grid(True)
        
        # Plot joint velocities
        if qvel:
            axes[0, 1].plot(qvel)
            axes[0, 1].set_title("Joint Velocities")
            axes[0, 1].set_xlabel("Step")
            axes[0, 1].set_ylabel("Velocity")
            axes[0, 1].grid(True)
        
        # Plot actions
        if actions:
            axes[1, 0].plot(actions)
            axes[1, 0].set_title("Actions")
            axes[1, 0].set_xlabel("Step")
            axes[1, 0].set_ylabel("Action")
            axes[1, 0].grid(True)
        
        # Plot rewards
        if rewards:
            axes[1, 1].plot(rewards)
            axes[1, 1].set_title("Rewards")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("Reward")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if out_file:
            plt.savefig(out_file)
            print(f"Plot saved to: {out_file}")
        else:
            plt.show()
    except Exception as e:
        print(f"Error plotting trajectory: {e}")


def main():
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("log_dir", type=str, help="Path to log directory")
    parser.add_argument("--out", type=str, default=str(PPO_PLOTS_DIR / "training_curves.png"), help="Output file")
    parser.add_argument("--type", type=str, choices=["curves", "trajectory"], 
                       default="curves", help="Plot type")
    
    args = parser.parse_args()

    # Redirect legacy paths and ensure directories
    out_path = Path(redirect_legacy_dir(args.out)) if args.out else None
    if out_path:
        ensure_dir_exists(out_path.parent)

    if args.type == "curves":
        plot_training_curves(args.log_dir, str(out_path) if out_path else None)
    else:
        print("Trajectory plotting not yet implemented")


if __name__ == "__main__":
    main()

