#!/usr/bin/env python3
"""
Comprehensive Model Comparison Report

Generate detailed comparison between baseline and enhanced models.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.models.models_torch import TorchMLPLarge


def load_model(checkpoint_path: str):
    """Load a trained BC model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    meta_path = checkpoint_path.replace('.pt', '_meta.json')
    import json
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


def evaluate_robustness(model, metadata, model_name: str):
    """Evaluate model robustness across different physics conditions."""
    print(f"\n=== Robustness Evaluation: {model_name} ===")
    
    physics_conditions = [
        ("nominal", {"ramp_deg": 10.0, "friction": 0.9}),
        ("gentle", {"ramp_deg": 8.0, "friction": 0.9}),
        ("steep", {"ramp_deg": 12.0, "friction": 0.9}),
        ("low_friction", {"ramp_deg": 10.0, "friction": 0.6}),
        ("high_friction", {"ramp_deg": 10.0, "friction": 1.0}),
    ]
    
    results = {}
    
    for condition_name, physics in physics_conditions:
        print(f"\nTesting {condition_name}: ramp={physics['ramp_deg']}°, friction={physics['friction']}")
        
        # Create environment with specific physics
        env = PassiveWalkerEnv(mode='fsm', ctrl_hz=100)
        env.ramp_deg = physics['ramp_deg']
        env.friction = physics['friction']
        
        episode_results = []
        
        for episode_id in range(3):  # 3 episodes per condition
            obs, _ = env.reset(seed=42 + episode_id)
            total_reward = 0.0
            steps = 0
            
            while env.data.time < 20.0 and steps < 2000:  # 20 second episodes
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = model(obs_tensor)
                    action = action_tensor.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            success = steps >= 1500  # Success if walked for >15s
            episode_results.append({
                'success': success,
                'total_reward': total_reward,
                'steps': steps,
                'duration': env.data.time
            })
        
        # Calculate metrics for this condition
        success_rate = np.mean([ep['success'] for ep in episode_results])
        avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
        avg_steps = np.mean([ep['steps'] for ep in episode_results])
        
        results[condition_name] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'episodes': len(episode_results)
        }
        
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
    
    return results


def evaluate_control_frequency(model, metadata, model_name: str):
    """Evaluate model performance at different control frequencies."""
    print(f"\n=== Control Frequency Evaluation: {model_name} ===")
    
    frequencies = [100, 150, 200]
    results = {}
    
    for freq in frequencies:
        print(f"\nTesting {freq}Hz control frequency")
        
        env = PassiveWalkerEnv(mode='fsm', ctrl_hz=freq)
        
        episode_results = []
        
        for episode_id in range(3):
            obs, _ = env.reset(seed=42 + episode_id)
            total_reward = 0.0
            steps = 0
            
            while env.data.time < 15.0 and steps < 1500:  # 15 second episodes
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = model(obs_tensor)
                    action = action_tensor.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            success = steps >= 1000  # Success if walked for >10s
            episode_results.append({
                'success': success,
                'total_reward': total_reward,
                'steps': steps,
                'duration': env.data.time
            })
        
        success_rate = np.mean([ep['success'] for ep in episode_results])
        avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
        avg_steps = np.mean([ep['steps'] for ep in episode_results])
        
        results[freq] = {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps
        }
        
        print(f"  Success Rate: {success_rate:.1%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
    
    return results


def main():
    """Main comparison evaluation."""
    print("=== Comprehensive BC Model Comparison ===")
    
    # Model paths
    baseline_path = "checkpoints/checkpoints_baseline/torch_both_seed123_ep1_steps180000.pt"
    enhanced_path = "checkpoints/checkpoints_enhanced/torch_both_seed123_ep1_steps18000.pt"
    
    all_results = {}
    
    # Evaluate baseline model
    if os.path.exists(baseline_path):
        print("\n" + "="*60)
        print("BASELINE MODEL EVALUATION")
        print("="*60)
        
        baseline_model, baseline_metadata = load_model(baseline_path)
        
        # Standard FSM evaluation
        print("\n=== Standard FSM Evaluation ===")
        env = PassiveWalkerEnv(mode='fsm', ctrl_hz=100)
        
        episode_results = []
        for episode_id in range(5):
            obs, _ = env.reset(seed=42 + episode_id)
            total_reward = 0.0
            steps = 0
            
            while env.data.time < 25.0 and steps < 2500:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = baseline_model(obs_tensor)
                    action = action_tensor.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            success = steps >= 2000
            episode_results.append({
                'success': success,
                'total_reward': total_reward,
                'steps': steps,
                'duration': env.data.time
            })
        
        baseline_standard = {
            'success_rate': np.mean([ep['success'] for ep in episode_results]),
            'avg_reward': np.mean([ep['total_reward'] for ep in episode_results]),
            'avg_steps': np.mean([ep['steps'] for ep in episode_results]),
            'avg_duration': np.mean([ep['duration'] for ep in episode_results])
        }
        
        print(f"Success Rate: {baseline_standard['success_rate']:.1%}")
        print(f"Average Reward: {baseline_standard['avg_reward']:.2f}")
        print(f"Average Steps: {baseline_standard['avg_steps']:.1f}")
        print(f"Average Duration: {baseline_standard['avg_duration']:.1f}s")
        
        # Robustness evaluation
        baseline_robustness = evaluate_robustness(baseline_model, baseline_metadata, "Baseline")
        
        # Control frequency evaluation
        baseline_frequency = evaluate_control_frequency(baseline_model, baseline_metadata, "Baseline")
        
        all_results['baseline'] = {
            'standard': baseline_standard,
            'robustness': baseline_robustness,
            'frequency': baseline_frequency
        }
    
    # Evaluate enhanced model
    if os.path.exists(enhanced_path):
        print("\n" + "="*60)
        print("ENHANCED MODEL EVALUATION")
        print("="*60)
        
        enhanced_model, enhanced_metadata = load_model(enhanced_path)
        
        # Standard FSM evaluation
        print("\n=== Standard FSM Evaluation ===")
        env = PassiveWalkerEnv(mode='fsm', ctrl_hz=100)
        
        episode_results = []
        for episode_id in range(5):
            obs, _ = env.reset(seed=42 + episode_id)
            total_reward = 0.0
            steps = 0
            
            while env.data.time < 25.0 and steps < 2500:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = enhanced_model(obs_tensor)
                    action = action_tensor.squeeze(0).numpy()
                
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            success = steps >= 2000
            episode_results.append({
                'success': success,
                'total_reward': total_reward,
                'steps': steps,
                'duration': env.data.time
            })
        
        enhanced_standard = {
            'success_rate': np.mean([ep['success'] for ep in episode_results]),
            'avg_reward': np.mean([ep['total_reward'] for ep in episode_results]),
            'avg_steps': np.mean([ep['steps'] for ep in episode_results]),
            'avg_duration': np.mean([ep['duration'] for ep in episode_results])
        }
        
        print(f"Success Rate: {enhanced_standard['success_rate']:.1%}")
        print(f"Average Reward: {enhanced_standard['avg_reward']:.2f}")
        print(f"Average Steps: {enhanced_standard['avg_steps']:.1f}")
        print(f"Average Duration: {enhanced_standard['avg_duration']:.1f}s")
        
        # Robustness evaluation
        enhanced_robustness = evaluate_robustness(enhanced_model, enhanced_metadata, "Enhanced")
        
        # Control frequency evaluation
        enhanced_frequency = evaluate_control_frequency(enhanced_model, enhanced_metadata, "Enhanced")
        
        all_results['enhanced'] = {
            'standard': enhanced_standard,
            'robustness': enhanced_robustness,
            'frequency': enhanced_frequency
        }
    
    # Print comprehensive comparison
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    if 'baseline' in all_results and 'enhanced' in all_results:
        baseline = all_results['baseline']
        enhanced = all_results['enhanced']
        
        print(f"\n{'Metric':<25} {'Baseline':<15} {'Enhanced':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Standard metrics
        print(f"{'Success Rate':<25} {baseline['standard']['success_rate']:<15.1%} {enhanced['standard']['success_rate']:<15.1%} {enhanced['standard']['success_rate'] - baseline['standard']['success_rate']:<+15.1%}")
        print(f"{'Avg Reward':<25} {baseline['standard']['avg_reward']:<15.2f} {enhanced['standard']['avg_reward']:<15.2f} {enhanced['standard']['avg_reward'] - baseline['standard']['avg_reward']:<+15.2f}")
        print(f"{'Avg Steps':<25} {baseline['standard']['avg_steps']:<15.1f} {enhanced['standard']['avg_steps']:<15.1f} {enhanced['standard']['avg_steps'] - baseline['standard']['avg_steps']:<+15.1f}")
        print(f"{'Avg Duration':<25} {baseline['standard']['avg_duration']:<15.1f} {enhanced['standard']['avg_duration']:<15.1f} {enhanced['standard']['avg_duration'] - baseline['standard']['avg_duration']:<+15.1f}")
        
        # Robustness comparison
        print(f"\n{'ROBUSTNESS COMPARISON':<25}")
        print(f"{'Condition':<25} {'Baseline':<15} {'Enhanced':<15} {'Difference':<15}")
        print("-" * 70)
        
        for condition in ['nominal', 'gentle', 'steep', 'low_friction', 'high_friction']:
            if condition in baseline['robustness'] and condition in enhanced['robustness']:
                baseline_success = baseline['robustness'][condition]['success_rate']
                enhanced_success = enhanced['robustness'][condition]['success_rate']
                diff = enhanced_success - baseline_success
                print(f"{condition:<25} {baseline_success:<15.1%} {enhanced_success:<15.1%} {diff:<+15.1%}")
        
        # Control frequency comparison
        print(f"\n{'CONTROL FREQUENCY COMPARISON':<25}")
        print(f"{'Frequency':<25} {'Baseline':<15} {'Enhanced':<15} {'Difference':<15}")
        print("-" * 70)
        
        for freq in [100, 150, 200]:
            if freq in baseline['frequency'] and freq in enhanced['frequency']:
                baseline_success = baseline['frequency'][freq]['success_rate']
                enhanced_success = enhanced['frequency'][freq]['success_rate']
                diff = enhanced_success - baseline_success
                print(f"{freq}Hz{'':<22} {baseline_success:<15.1%} {enhanced_success:<15.1%} {diff:<+15.1%}")
        
        # Overall assessment
        print(f"\n{'OVERALL ASSESSMENT':<25}")
        print("=" * 50)
        
        if enhanced['standard']['success_rate'] >= baseline['standard']['success_rate']:
            print("✅ Enhanced model maintains or improves success rate")
        else:
            print("❌ Enhanced model shows reduced success rate")
        
        if enhanced['standard']['avg_reward'] >= baseline['standard']['avg_reward']:
            print("✅ Enhanced model maintains or improves reward")
        else:
            print("❌ Enhanced model shows reduced reward")
        
        # Check robustness
        baseline_robustness_avg = np.mean([baseline['robustness'][cond]['success_rate'] for cond in baseline['robustness']])
        enhanced_robustness_avg = np.mean([enhanced['robustness'][cond]['success_rate'] for cond in enhanced['robustness']])
        
        if enhanced_robustness_avg >= baseline_robustness_avg:
            print("✅ Enhanced model shows improved robustness")
        else:
            print("❌ Enhanced model shows reduced robustness")
    
    return all_results


if __name__ == "__main__":
    results = main()

