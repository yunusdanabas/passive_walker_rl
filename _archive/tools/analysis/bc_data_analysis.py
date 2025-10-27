#!/usr/bin/env python3
"""
Analyze and replay collected FSM demonstration data with detailed visualizations.
"""
import os
import numpy as np
import argparse
import json
from pathlib import Path
from passive_walker.core.env import PassiveWalkerEnv

# Try to import matplotlib, fallback to text-only if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, generating text-only analysis")


def analyze_dataset(data_dir, output_dir="analysis", save_json=False):
    """Analyze a dataset and generate detailed visualizations."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all episode files
    episode_files = sorted(data_dir.glob("episode_*.npz"))
    if not episode_files:
        print(f"No episode files found in {data_dir}")
        return
    
    print(f"Analyzing {len(episode_files)} episodes from {data_dir}")
    
    # Load metadata
    meta_file = data_dir / "meta.json"
    meta = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"Dataset: {meta.get('episodes', 'unknown')} episodes, "
              f"{meta.get('target_duration_sec', 'unknown')}s each")
    
    # Analyze episodes
    all_distances = []
    all_rewards = []
    all_pitches = []
    all_heights = []
    episode_lengths = []
    fall_count = 0
    
    # Detailed episode data for visualization
    episode_data = []
    
    for ep_file in episode_files:
        data = np.load(ep_file)
        
        # Extract metrics
        distance = np.sum(data['info_dx'])
        avg_reward = np.mean(data['rew'])
        max_pitch = np.max(data['info_pitch'])
        min_height = np.min(data['info_torso_z'])
        length = len(data['obs']) / 100  # seconds
        
        # Check for falls (episodes that ended early)
        if length < 24.0:  # Less than 24 seconds indicates early termination
            fall_count += 1
        
        all_distances.append(distance)
        all_rewards.append(avg_reward)
        all_pitches.append(max_pitch)
        all_heights.append(min_height)
        episode_lengths.append(length)
        
        # Store detailed data for first few episodes
        if len(episode_data) < 5:
            episode_data.append({
                'file': ep_file.name,
                'time': np.arange(len(data['info_dx'])) / 100,
                'dx': data['info_dx'],
                'pitch': data['info_pitch'],
                'height': data['info_torso_z'],
                'reward': data['rew'],
                'fsm_hip': data['info_fsm_hip'],
                'fsm_k1': data['info_fsm_k1'],
                'fsm_k2': data['info_fsm_k2'],
                'qdes': data['info_qdes'],
                'u_abs': data['info_u_abs_sum']
            })
    
    # Print summary statistics
    dataset_name = data_dir.name
    print(f"\n=== {dataset_name.upper()} DATASET ANALYSIS ===")
    print(f"Total episodes: {len(episode_files)}")
    print(f"Episodes with falls: {fall_count} ({fall_count/len(episode_files)*100:.1f}%)")
    print(f"Success rate: {(len(episode_files)-fall_count)/len(episode_files)*100:.1f}%")
    print()
    
    print("DURATION STATISTICS:")
    print(f"  Average: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} seconds")
    print(f"  Range: {np.min(episode_lengths):.1f} - {np.max(episode_lengths):.1f} seconds")
    print(f"  Target: 25.0 seconds")
    print()
    
    print("DISTANCE STATISTICS:")
    print(f"  Average: {np.mean(all_distances):.1f} ± {np.std(all_distances):.1f} meters")
    print(f"  Range: {np.min(all_distances):.1f} - {np.max(all_distances):.1f} meters")
    print(f"  Average speed: {np.mean(all_distances)/25:.3f} m/s")
    print()
    
    print("REWARD STATISTICS:")
    print(f"  Average: {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"  Range: {np.min(all_rewards):.3f} - {np.max(all_rewards):.3f}")
    print()
    
    print("STABILITY STATISTICS:")
    print(f"  Max pitch: {np.mean(all_pitches):.3f} ± {np.std(all_pitches):.3f} rad")
    print(f"  Pitch range: {np.min(all_pitches):.3f} - {np.max(all_pitches):.3f} rad")
    print(f"  Min height: {np.mean(all_heights):.3f} ± {np.std(all_heights):.3f} m")
    print(f"  Height range: {np.min(all_heights):.3f} - {np.max(all_heights):.3f} m")
    print()
    
    # Quality assessment
    quality_score = 0
    if np.mean(episode_lengths) >= 24.5:
        quality_score += 1
        print("✅ Duration: Excellent (≥24.5s average)")
    elif np.mean(episode_lengths) >= 20.0:
        quality_score += 0.5
        print("⚠️  Duration: Good (≥20s average)")
    else:
        print("❌ Duration: Poor (<20s average)")
    
    if fall_count / len(episode_files) <= 0.05:
        quality_score += 1
        print("✅ Stability: Excellent (≤5% fall rate)")
    elif fall_count / len(episode_files) <= 0.20:
        quality_score += 0.5
        print("⚠️  Stability: Good (≤20% fall rate)")
    else:
        print("❌ Stability: Poor (>20% fall rate)")
    
    if np.mean(all_distances) >= 30.0:
        quality_score += 1
        print("✅ Distance: Excellent (≥30m average)")
    elif np.mean(all_distances) >= 20.0:
        quality_score += 0.5
        print("⚠️  Distance: Good (≥20m average)")
    else:
        print("❌ Distance: Poor (<20m average)")
    
    print(f"\nOVERALL QUALITY SCORE: {quality_score}/3.0")
    if quality_score >= 2.5:
        print("🎉 Dataset quality: EXCELLENT - Ready for training!")
    elif quality_score >= 2.0:
        print("✅ Dataset quality: GOOD - Ready for training")
    elif quality_score >= 1.5:
        print("⚠️  Dataset quality: FAIR - Consider collecting more data")
    else:
        print("❌ Dataset quality: POOR - Need better data collection")
    
    # Generate visualizations if matplotlib is available
    if HAS_MATPLOTLIB and episode_data:
        generate_visualizations(dataset_name, episode_data, 
                              all_distances, all_rewards, all_pitches, all_heights, 
                              episode_lengths, output_dir)
    else:
        print(f"\nNote: Install matplotlib for PNG visualizations: pip install matplotlib")
        print(f"Text-only analysis completed for {dataset_name}")
    
    # Save detailed analysis to JSON (optional)
    if save_json:
        analysis_summary = {
            "dataset_name": dataset_name,
            "total_episodes": len(episode_files),
            "fall_count": fall_count,
            "success_rate": (len(episode_files)-fall_count)/len(episode_files)*100,
            "duration_stats": {
                "mean": float(np.mean(episode_lengths)),
                "std": float(np.std(episode_lengths)),
                "min": float(np.min(episode_lengths)),
                "max": float(np.max(episode_lengths))
            },
            "distance_stats": {
                "mean": float(np.mean(all_distances)),
                "std": float(np.std(all_distances)),
                "min": float(np.min(all_distances)),
                "max": float(np.max(all_distances)),
                "avg_speed": float(np.mean(all_distances)/25)
            },
            "reward_stats": {
                "mean": float(np.mean(all_rewards)),
                "std": float(np.std(all_rewards)),
                "min": float(np.min(all_rewards)),
                "max": float(np.max(all_rewards))
            },
            "stability_stats": {
                "max_pitch_mean": float(np.mean(all_pitches)),
                "max_pitch_std": float(np.std(all_pitches)),
                "min_height_mean": float(np.mean(all_heights)),
                "min_height_std": float(np.std(all_heights))
            },
            "quality_score": quality_score,
            "metadata": meta
        }
        
        summary_file = output_dir / f"{dataset_name}_analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        print(f"\nDetailed analysis saved to: {summary_file}")


def generate_visualizations(dataset_name, episode_data, distances, rewards, pitches, heights, lengths, output_dir):
    """Generate detailed visualization plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Episode Overview Dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name.upper()} Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Episode lengths histogram
    axes[0,0].hist(lengths, bins=15, alpha=0.7, color=colors[0], edgecolor='black')
    axes[0,0].set_title('Episode Duration Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Duration (seconds)')
    axes[0,0].set_ylabel('Count')
    axes[0,0].axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(lengths):.1f}s')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Distance histogram
    axes[0,1].hist(distances, bins=15, alpha=0.7, color=colors[1], edgecolor='black')
    axes[0,1].set_title('Total Distance Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Distance (meters)')
    axes[0,1].set_ylabel('Count')
    axes[0,1].axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(distances):.1f}m')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Reward histogram
    axes[0,2].hist(rewards, bins=15, alpha=0.7, color=colors[2], edgecolor='black')
    axes[0,2].set_title('Average Reward Distribution', fontweight='bold')
    axes[0,2].set_xlabel('Reward')
    axes[0,2].set_ylabel('Count')
    axes[0,2].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(rewards):.3f}')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Pitch stability
    axes[1,0].hist(pitches, bins=15, alpha=0.7, color=colors[3], edgecolor='black')
    axes[1,0].set_title('Max Pitch Distribution', fontweight='bold')
    axes[1,0].set_xlabel('Max Pitch (rad)')
    axes[1,0].set_ylabel('Count')
    axes[1,0].axvline(np.mean(pitches), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(pitches):.3f}')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Height stability
    axes[1,1].hist(heights, bins=15, alpha=0.7, color=colors[4], edgecolor='black')
    axes[1,1].set_title('Min Torso Height Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Min Height (m)')
    axes[1,1].set_ylabel('Count')
    axes[1,1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {np.mean(heights):.3f}m')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Quality score visualization
    quality_categories = ['Duration', 'Stability', 'Distance']
    quality_values = [
        1 if np.mean(lengths) >= 24.5 else 0.5 if np.mean(lengths) >= 20.0 else 0,
        1 if len([l for l in lengths if l < 24.0])/len(lengths) <= 0.05 else 0.5 if len([l for l in lengths if l < 24.0])/len(lengths) <= 0.20 else 0,
        1 if np.mean(distances) >= 30.0 else 0.5 if np.mean(distances) >= 20.0 else 0
    ]
    
    bars = axes[1,2].bar(quality_categories, quality_values, color=['green' if v >= 1 else 'orange' if v >= 0.5 else 'red' for v in quality_values])
    axes[1,2].set_title('Quality Assessment', fontweight='bold')
    axes[1,2].set_ylabel('Quality Score')
    axes[1,2].set_ylim(0, 1.2)
    axes[1,2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, quality_values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    dashboard_path = output_dir / f"{dataset_name}_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dashboard plot saved: {dashboard_path}")
    
    # 2. Detailed Episode Analysis
    if len(episode_data) >= 3:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{dataset_name.upper()} - Detailed Episode Analysis', fontsize=16, fontweight='bold')
        
        for i, ep in enumerate(episode_data[:3]):
            color = colors[i % len(colors)]
            
            # Forward velocity
            axes[i, 0].plot(ep['time'], ep['dx'], color=color, linewidth=1.5, label=f'Episode {i+1}')
            axes[i, 0].set_title(f'Episode {i+1}: Forward Velocity', fontweight='bold')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Velocity (m/s)')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].legend()
            
            # Pitch angle
            axes[i, 1].plot(ep['time'], ep['pitch'], color=color, linewidth=1.5, label=f'Episode {i+1}')
            axes[i, 1].set_title(f'Episode {i+1}: Pitch Angle', fontweight='bold')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Pitch (rad)')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].legend()
        
        plt.tight_layout()
        episodes_path = output_dir / f"{dataset_name}_episodes_detail.png"
        plt.savefig(episodes_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Episode details plot saved: {episodes_path}")
    
    # 3. FSM State Analysis
    if episode_data:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{dataset_name.upper()} - FSM State Analysis', fontsize=16, fontweight='bold')
        
        ep = episode_data[0]  # Use first episode
        
        # FSM Hip States
        axes[0, 0].plot(ep['time'], ep['fsm_hip'], 'b-', linewidth=2, label='Hip State')
        axes[0, 0].set_title('FSM Hip State Transitions', fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('State')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # FSM Knee States
        axes[0, 1].plot(ep['time'], ep['fsm_k1'], 'r-', linewidth=2, label='Left Knee')
        axes[0, 1].plot(ep['time'], ep['fsm_k2'], 'g-', linewidth=2, label='Right Knee')
        axes[0, 1].set_title('FSM Knee State Transitions', fontweight='bold')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('State')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Control Effort
        axes[1, 0].plot(ep['time'], ep['u_abs'], 'purple', linewidth=2, label='Control Effort')
        axes[1, 0].set_title('Control Effort Over Time', fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('|u| (N⋅m)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Joint Desired Positions
        axes[1, 1].plot(ep['time'], ep['qdes'][:, 0], 'b-', linewidth=2, label='Hip')
        axes[1, 1].plot(ep['time'], ep['qdes'][:, 1], 'r-', linewidth=2, label='Left Knee')
        axes[1, 1].plot(ep['time'], ep['qdes'][:, 2], 'g-', linewidth=2, label='Right Knee')
        axes[1, 1].set_title('Desired Joint Positions', fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Position (rad)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        fsm_path = output_dir / f"{dataset_name}_fsm_analysis.png"
        plt.savefig(fsm_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"FSM analysis plot saved: {fsm_path}")
    
    # 4. Performance Metrics Correlation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dataset_name.upper()} - Performance Correlations', fontsize=16, fontweight='bold')
    
    # Distance vs Duration
    axes[0, 0].scatter(lengths, distances, alpha=0.6, s=50, color=colors[0])
    axes[0, 0].set_xlabel('Duration (s)')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Distance vs Duration')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward vs Distance
    axes[0, 1].scatter(distances, rewards, alpha=0.6, s=50, color=colors[1])
    axes[0, 1].set_xlabel('Distance (m)')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Reward vs Distance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Pitch vs Height
    axes[1, 0].scatter(pitches, heights, alpha=0.6, s=50, color=colors[2])
    axes[1, 0].set_xlabel('Max Pitch (rad)')
    axes[1, 0].set_ylabel('Min Height (m)')
    axes[1, 0].set_title('Pitch vs Height Stability')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode Index vs Performance
    episode_indices = range(len(distances))
    axes[1, 1].plot(episode_indices, distances, 'o-', alpha=0.7, color=colors[3], label='Distance')
    ax2 = axes[1, 1].twinx()
    ax2.plot(episode_indices, rewards, 's-', alpha=0.7, color=colors[4], label='Reward')
    axes[1, 1].set_xlabel('Episode Index')
    axes[1, 1].set_ylabel('Distance (m)', color=colors[3])
    ax2.set_ylabel('Average Reward', color=colors[4])
    axes[1, 1].set_title('Performance Over Episodes')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    correlation_path = output_dir / f"{dataset_name}_correlations.png"
    plt.savefig(correlation_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Correlation analysis plot saved: {correlation_path}")


def replay_episode(episode_path, speed=1.0):
    """Replay a collected episode with GUI."""
    
    # Load episode data
    data = np.load(episode_path)
    print(f"Loading episode: {episode_path}")
    print(f"Duration: {len(data['obs']) / 100:.1f}s")
    print(f"Steps: {len(data['obs'])}")
    
    # Create environment with GUI
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    
    # Reset environment
    obs, _ = env.reset(seed=int(data['seed']))
    
    print("Starting episode replay...")
    print("Press Ctrl+C to stop")
    
    try:
        for t in range(len(data['act'])):
            # Get the action from the episode (should be zeros for FSM mode)
            action = data['act'][t]
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Print progress every 100 steps
            if t % 100 == 0:
                print(f"t={t:4d}/{len(data['act'])} "
                      f"dx={info.get('dx', 0):+.4f} "
                      f"pitch={info.get('pitch_abs', 0):.3f} "
                      f"reward={reward:+.4f}")
            
            # Control playback speed
            if speed != 1.0:
                import time
                sleep_time = max(0, (1.0/speed - 1.0) * 0.01)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            if done:
                print(f"Episode ended early at step {t}")
                break
                
    except KeyboardInterrupt:
        print("\nReplay interrupted by user")
    
    finally:
        env.close()
        print("Episode replay completed")


def main():
    parser = argparse.ArgumentParser("Analyze and replay collected FSM data")
    parser.add_argument("data_dir", help="Directory containing episode_*.npz files")
    parser.add_argument("--replay", help="Replay specific episode file")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--output", default="analysis", help="Output directory for plots and analysis")
    parser.add_argument("--save-json", action="store_true", help="Save detailed analysis as JSON")
    args = parser.parse_args()
    
    if args.replay:
        replay_episode(args.replay, args.speed)
    else:
        analyze_dataset(args.data_dir, args.output, args.save_json)


if __name__ == "__main__":
    main()