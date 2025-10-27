#!/usr/bin/env python3
"""
Comprehensive Model Evaluation & Visualization

Evaluate all trained models and generate comprehensive plots and analysis.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.models.models_torch import TorchMLPLarge


def load_model(checkpoint_path: str):
    """Load a trained BC model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    meta_path = checkpoint_path.replace('.pt', '_meta.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    model = TorchMLPLarge(
        in_dim=metadata['input_dim'],
        out_dim=metadata['output_dim'],
        hidden=512,
        dropout=0.1
    )
    
    model.load_state_dict(checkpoint)
    model.eval()
    return model, metadata


def evaluate_model_comprehensive(model, metadata, model_name: str, physics_conditions: list, episodes_per_condition: int = 3):
    """Comprehensive evaluation across multiple physics conditions."""
    print(f"\n=== Comprehensive Evaluation: {model_name} ===")
    
    results = {}
    
    for condition_name, physics in physics_conditions:
        print(f"\nTesting {condition_name}: ramp={physics['ramp_deg']}°, friction={physics['friction']}")
        
        # Create environment
        env = PassiveWalkerEnv(mode='fsm', ctrl_hz=100)
        env.ramp_deg = physics['ramp_deg']
        env.friction = physics['friction']
        
        episode_data = []
        
        for episode_id in range(episodes_per_condition):
            obs, _ = env.reset(seed=42 + episode_id)
            total_reward = 0.0
            steps = 0
            
            # Collect trajectory data
            observations = []
            actions = []
            rewards = []
            joint_positions = []
            joint_velocities = []
            
            while env.data.time < 20.0 and steps < 2000:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = model(obs_tensor)
                    action = action_tensor.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                
                # Store trajectory data
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(reward)
                joint_positions.append([
                    env.data.qpos[env.qpos_hip],
                    env.data.qpos[env.qpos_lk],
                    env.data.qpos[env.qpos_rk]
                ])
                joint_velocities.append([
                    env.data.qvel[env.qvel_hip],
                    env.data.qvel[env.qvel_lk],
                    env.data.qvel[env.qvel_rk]
                ])
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            success = steps >= 1500
            episode_data.append({
                'success': success,
                'total_reward': total_reward,
                'steps': steps,
                'duration': env.data.time,
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'joint_positions': joint_positions,
                'joint_velocities': joint_velocities
            })
        
        # Calculate metrics
        success_rate = np.mean([ep['success'] for ep in episode_data])
        avg_reward = np.mean([ep['total_reward'] for ep in episode_data])
        avg_steps = np.mean([ep['steps'] for ep in episode_data])
        avg_duration = np.mean([ep['duration'] for ep in episode_data])
        
        results[condition_name] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'avg_duration': avg_duration,
            'episode_data': episode_data
        }
        
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Average Duration: {avg_duration:.1f}s")
    
    return results


def create_comparison_plots(all_results, output_dir: str = "outputs/evaluation_plots"):
    """Create comprehensive comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Success Rate Comparison
    plt.figure(figsize=(12, 8))
    
    models = list(all_results.keys())
    conditions = list(all_results[models[0]].keys())
    
    x = np.arange(len(conditions))
    width = 0.15
    
    for i, model in enumerate(models):
        success_rates = [all_results[model][cond]['success_rate'] for cond in conditions]
        plt.bar(x + i * width, success_rates, width, label=model, alpha=0.8)
    
    plt.xlabel('Physics Condition')
    plt.ylabel('Success Rate')
    plt.title('Model Success Rate Comparison Across Physics Conditions')
    plt.xticks(x + width * (len(models) - 1) / 2, conditions, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/success_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Reward Comparison
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        avg_rewards = [all_results[model][cond]['avg_reward'] for cond in conditions]
        plt.bar(x + i * width, avg_rewards, width, label=model, alpha=0.8)
    
    plt.xlabel('Physics Condition')
    plt.ylabel('Average Reward')
    plt.title('Model Reward Comparison Across Physics Conditions')
    plt.xticks(x + width * (len(models) - 1) / 2, conditions, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Robustness Heatmap
    plt.figure(figsize=(10, 6))
    
    success_matrix = []
    for model in models:
        success_rates = [all_results[model][cond]['success_rate'] for cond in conditions]
        success_matrix.append(success_rates)
    
    success_matrix = np.array(success_matrix)
    
    im = plt.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='Success Rate')
    
    plt.yticks(range(len(models)), models)
    plt.xticks(range(len(conditions)), conditions, rotation=45)
    plt.title('Robustness Heatmap: Success Rate by Model and Condition')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(conditions)):
            plt.text(j, i, f'{success_matrix[i, j]:.1%}', ha='center', va='center', color='black')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/robustness_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Trajectory Comparison (for nominal condition)
    if 'nominal' in conditions:
        num_models = len(models)
        rows = (num_models + 1) // 2  # Round up for odd numbers
        cols = 2
        
        plt.figure(figsize=(15, 5 * rows))
        
        # Plot joint trajectories for each model
        for i, model in enumerate(models):
            if 'nominal' in all_results[model]:
                episode_data = all_results[model]['nominal']['episode_data']
                if episode_data and episode_data[0]['success']:  # Use successful episode
                    episode = episode_data[0]
                    joint_positions = np.array(episode['joint_positions'])
                    time_steps = np.arange(len(joint_positions)) * 0.01
                    
                    plt.subplot(rows, cols, i + 1)
                    plt.plot(time_steps, joint_positions[:, 0], label='Hip', alpha=0.8)
                    plt.plot(time_steps, joint_positions[:, 1], label='Left Knee', alpha=0.8)
                    plt.plot(time_steps, joint_positions[:, 2], label='Right Knee', alpha=0.8)
                    plt.title(f'{model} - Joint Trajectories')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Joint Angle (rad)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/trajectory_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nPlots saved to: {output_dir}/")


def create_detailed_report(all_results, output_dir: str = "outputs/evaluation_reports"):
    """Create detailed evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{output_dir}/evaluation_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write("# BC Model Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        
        # Calculate overall metrics
        models = list(all_results.keys())
        conditions = list(all_results[models[0]].keys())
        
        f.write("| Model | Overall Success Rate | Avg Reward | Best Condition | Worst Condition |\n")
        f.write("|-------|---------------------|------------|----------------|-----------------|\n")
        
        for model in models:
            success_rates = [all_results[model][cond]['success_rate'] for cond in conditions]
            avg_rewards = [all_results[model][cond]['avg_reward'] for cond in conditions]
            
            overall_success = np.mean(success_rates)
            overall_reward = np.mean(avg_rewards)
            
            best_condition = conditions[np.argmax(success_rates)]
            worst_condition = conditions[np.argmin(success_rates)]
            
            f.write(f"| {model} | {overall_success:.1%} | {overall_reward:.2f} | {best_condition} | {worst_condition} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for model in models:
            f.write(f"### {model}\n\n")
            
            f.write("| Condition | Success Rate | Avg Reward | Avg Steps | Avg Duration |\n")
            f.write("|-----------|--------------|------------|-----------|--------------|\n")
            
            for condition in conditions:
                result = all_results[model][condition]
                f.write(f"| {condition} | {result['success_rate']:.1%} | {result['avg_reward']:.2f} | {result['avg_steps']:.1f} | {result['avg_duration']:.1f}s |\n")
            
            f.write("\n")
        
        f.write("## Analysis\n\n")
        
        # Find best performing model
        model_scores = {}
        for model in models:
            success_rates = [all_results[model][cond]['success_rate'] for cond in conditions]
            avg_rewards = [all_results[model][cond]['avg_reward'] for cond in conditions]
            model_scores[model] = np.mean(success_rates) * np.mean(avg_rewards)
        
        best_model = max(model_scores, key=model_scores.get)
        f.write(f"**Best Overall Model:** {best_model}\n\n")
        
        # Condition analysis
        f.write("### Condition Analysis\n\n")
        f.write("| Condition | Best Model | Success Rate | Notes |\n")
        f.write("|-----------|------------|--------------|-------|\n")
        
        for condition in conditions:
            best_model_for_condition = max(models, key=lambda m: all_results[m][condition]['success_rate'])
            best_success_rate = all_results[best_model_for_condition][condition]['success_rate']
            
            notes = []
            if best_success_rate < 0.5:
                notes.append("Challenging condition")
            elif best_success_rate > 0.9:
                notes.append("Easy condition")
            
            f.write(f"| {condition} | {best_model_for_condition} | {best_success_rate:.1%} | {', '.join(notes) if notes else 'Normal'} |\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Generate recommendations
        f.write("1. **For PPO Training:** Use the best performing model as initialization\n")
        f.write("2. **For Robustness:** Focus on conditions where models struggle\n")
        f.write("3. **For Data Collection:** Collect more data for challenging conditions\n")
        f.write("4. **For Evaluation:** Use comprehensive physics condition testing\n\n")
    
    print(f"Report saved to: {report_path}")


def main():
    """Main evaluation and analysis."""
    print("=== Comprehensive BC Model Evaluation & Analysis ===")
    
    # Define physics conditions to test
    physics_conditions = [
        ("nominal", {"ramp_deg": 10.0, "friction": 0.9}),
        ("gentle", {"ramp_deg": 8.0, "friction": 0.9}),
        ("steep", {"ramp_deg": 12.0, "friction": 0.9}),
        ("low_friction", {"ramp_deg": 10.0, "friction": 0.6}),
        ("high_friction", {"ramp_deg": 10.0, "friction": 1.0}),
        ("gentle_low", {"ramp_deg": 8.0, "friction": 0.6}),
    ]
    
    # Model paths and names
    models_to_evaluate = [
        ("experiments/models/torch_hip_seed123_ep1_steps9000.pt", "Hip Control"),
        ("experiments/models/torch_both_seed456_ep1_steps9000.pt", "Both Joints"),
    ]
    
    all_results = {}
    
    # Evaluate each model
    for checkpoint_path, model_name in models_to_evaluate:
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"EVALUATING {model_name.upper()}")
            print(f"{'='*60}")
            
            model, metadata = load_model(checkpoint_path)
            results = evaluate_model_comprehensive(model, metadata, model_name, physics_conditions, episodes_per_condition=3)
            all_results[model_name] = results
        else:
            print(f"Model not found: {checkpoint_path}")
    
    # Generate plots and reports
    if all_results:
        print(f"\n{'='*60}")
        print("GENERATING ANALYSIS")
        print(f"{'='*60}")
        
        create_comparison_plots(all_results)
        create_detailed_report(all_results)
        
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Model':<15} {'Overall Success':<15} {'Avg Reward':<12} {'Best Condition':<15}")
        print("-" * 60)
        
        for model_name, results in all_results.items():
            condition_names = [cond[0] for cond in physics_conditions]
            success_rates = [results[cond]['success_rate'] for cond in condition_names]
            avg_rewards = [results[cond]['avg_reward'] for cond in condition_names]
            
            overall_success = np.mean(success_rates)
            overall_reward = np.mean(avg_rewards)
            best_condition = condition_names[np.argmax(success_rates)]
            
            print(f"{model_name:<15} {overall_success:<15.1%} {overall_reward:<12.2f} {best_condition:<15}")
    
    return all_results


if __name__ == "__main__":
    results = main()
