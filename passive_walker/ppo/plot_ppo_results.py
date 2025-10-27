#!/usr/bin/env python3
"""
Simple PPO Results Plotter

Reads TensorBoard logs and creates matplotlib plots.
No localhost required - saves plots as images.
"""

import argparse
from pathlib import Path
from passive_walker.config.paths import PPO_RUNS_DIR, PPO_PLOTS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def read_tensorboard_logs(log_dir):
    """Read tensorboard event files and extract metrics."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        print("Error: tensorboard not installed")
        print("Install with: pip install tensorboard")
        return None
    
    metrics = defaultdict(lambda: {'steps': [], 'values': []})
    
    # Find all event files
    event_files = list(Path(log_dir).rglob('events.out.tfevents.*'))
    
    if not event_files:
        print(f"No tensorboard event files found in {log_dir}")
        return None
    
    print(f"Found {len(event_files)} event file(s)")
    
    for event_file in event_files:
        print(f"Reading: {event_file.name}")
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        
        # Get all scalar tags
        tags = ea.Tags()['scalars']
        
        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                metrics[tag]['steps'].append(event.step)
                metrics[tag]['values'].append(event.value)
    
    return metrics


def plot_training_results(metrics, output_dir):
    """Create plots for training metrics."""
    if not metrics:
        print("No metrics to plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group metrics by category
    eval_metrics = {k: v for k, v in metrics.items() if k.startswith('eval/')}
    train_metrics = {k: v for k, v in metrics.items() if k.startswith('train/')}
    env_metrics = {k: v for k, v in metrics.items() if k.startswith('env/')}
    
    # 1. Evaluation Metrics
    if eval_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Return
        if 'eval/return' in eval_metrics:
            steps = eval_metrics['eval/return']['steps']
            values = eval_metrics['eval/return']['values']
            axes[0].plot(steps, values, 'b-', linewidth=2, label='Eval Return')
            
            if 'eval/return_std' in eval_metrics:
                std = eval_metrics['eval/return_std']['values']
                axes[0].fill_between(steps, 
                                    np.array(values) - np.array(std),
                                    np.array(values) + np.array(std),
                                    alpha=0.3, color='blue')
            
            axes[0].set_xlabel('Timesteps', fontsize=12)
            axes[0].set_ylabel('Return', fontsize=12)
            axes[0].set_title('Evaluation Return Over Time', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # Episode Length
        if 'eval/length' in eval_metrics:
            steps = eval_metrics['eval/length']['steps']
            values = eval_metrics['eval/length']['values']
            axes[1].plot(steps, values, 'g-', linewidth=2, label='Eval Length')
            
            if 'eval/length_std' in eval_metrics:
                std = eval_metrics['eval/length_std']['values']
                axes[1].fill_between(steps,
                                    np.array(values) - np.array(std),
                                    np.array(values) + np.array(std),
                                    alpha=0.3, color='green')
            
            axes[1].set_xlabel('Timesteps', fontsize=12)
            axes[1].set_ylabel('Episode Length', fontsize=12)
            axes[1].set_title('Episode Length Over Time', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        
        plt.tight_layout()
        eval_plot = os.path.join(output_dir, 'evaluation_metrics.png')
        plt.savefig(eval_plot, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {eval_plot}")
        plt.close()
    
    # 2. Training Metrics
    if train_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Policy Loss
        if 'train/policy_loss' in train_metrics:
            steps = train_metrics['train/policy_loss']['steps']
            values = train_metrics['train/policy_loss']['values']
            axes[0, 0].plot(steps, values, 'r-', linewidth=1.5)
            axes[0, 0].set_xlabel('Timesteps')
            axes[0, 0].set_ylabel('Policy Loss')
            axes[0, 0].set_title('Policy Loss', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Value Loss
        if 'train/value_loss' in train_metrics:
            steps = train_metrics['train/value_loss']['steps']
            values = train_metrics['train/value_loss']['values']
            axes[0, 1].plot(steps, values, 'b-', linewidth=1.5)
            axes[0, 1].set_xlabel('Timesteps')
            axes[0, 1].set_ylabel('Value Loss')
            axes[0, 1].set_title('Value Loss', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy
        if 'train/entropy_loss' in train_metrics:
            steps = train_metrics['train/entropy_loss']['steps']
            values = train_metrics['train/entropy_loss']['values']
            axes[1, 0].plot(steps, values, 'g-', linewidth=1.5)
            axes[1, 0].set_xlabel('Timesteps')
            axes[1, 0].set_ylabel('Entropy Loss')
            axes[1, 0].set_title('Entropy Loss', fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'train/learning_rate' in train_metrics:
            steps = train_metrics['train/learning_rate']['steps']
            values = train_metrics['train/learning_rate']['values']
            axes[1, 1].plot(steps, values, 'm-', linewidth=1.5)
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        train_plot = os.path.join(output_dir, 'training_metrics.png')
        plt.savefig(train_plot, dpi=150, bbox_inches='tight')
        print(f"✅ Saved: {train_plot}")
        plt.close()
    
    # 3. Summary Statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    if 'eval/return' in metrics:
        returns = metrics['eval/return']['values']
        print(f"\nEvaluation Returns:")
        print(f"  Initial:  {returns[0]:>8.2f}")
        print(f"  Final:    {returns[-1]:>8.2f}")
        print(f"  Best:     {max(returns):>8.2f}")
        print(f"  Mean:     {np.mean(returns):>8.2f}")
        print(f"  Std:      {np.std(returns):>8.2f}")
    
    if 'eval/length' in metrics:
        lengths = metrics['eval/length']['values']
        print(f"\nEpisode Lengths:")
        print(f"  Initial:  {lengths[0]:>8.1f}")
        print(f"  Final:    {lengths[-1]:>8.1f}")
        print(f"  Max:      {max(lengths):>8.1f}")
        print(f"  Mean:     {np.mean(lengths):>8.1f}")
    
    print("\n" + "="*70)
    print(f"\nPlots saved to: {output_dir}/")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training results")
    parser.add_argument("--logdir", type=str, default=str(PPO_RUNS_DIR),
                       help="Directory containing tensorboard logs")
    parser.add_argument("--output", type=str, default=str(PPO_PLOTS_DIR),
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PPO RESULTS PLOTTER")
    print(f"{'='*70}\n")
    print(f"Log directory: {args.logdir}")
    print(f"Output directory: {args.output}\n")
    
    # Redirect legacy paths and ensure dirs
    args.logdir = str(redirect_legacy_dir(args.logdir))
    args.output = str(redirect_legacy_dir(args.output))
    ensure_dir_exists(Path(args.output))

    # Read metrics
    metrics = read_tensorboard_logs(args.logdir)
    
    if metrics:
        # Create plots
        plot_training_results(metrics, args.output)
        print(f"✅ Done! Check {args.output}/ for plots.\n")
    else:
        print("❌ No data found. Run training first.\n")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

