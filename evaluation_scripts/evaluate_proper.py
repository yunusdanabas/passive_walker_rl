#!/usr/bin/env python3
"""
Proper Model Evaluation Script

Actually load and use the trained BC models for evaluation.
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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load metadata
    meta_path = checkpoint_path.replace('.pt', '_meta.json')
    import json
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = TorchMLPLarge(
        in_dim=metadata['input_dim'],
        out_dim=metadata['output_dim'],
        hidden=512,
        dropout=0.1
    )
    
    # Load weights
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, metadata


def evaluate_model_with_actual_model(checkpoint_path: str, model_name: str, episodes: int = 5, mode: str = 'fsm'):
    """Evaluate model using actual trained model predictions."""
    print(f"\n=== Evaluating {model_name} ({mode} mode) ===")
    
    # Load model
    model, metadata = load_model(checkpoint_path)
    print(f"Model loaded: {metadata['input_dim']}D input, {metadata['output_dim']}D output")
    
    # Create environment
    env = PassiveWalkerEnv(mode=mode, ctrl_hz=100)
    
    episode_results = []
    
    for episode_id in range(episodes):
        obs, _ = env.reset(seed=42 + episode_id)
        total_reward = 0.0
        steps = 0
        
        while env.data.time < 25.0 and steps < 2500:
            # Get model prediction
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_tensor = model(obs_tensor)
                action = action_tensor.squeeze(0).numpy()
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        success = steps >= 2000  # Consider success if walked for >20s
        episode_results.append({
            'episode_id': episode_id,
            'success': success,
            'total_reward': total_reward,
            'steps': steps,
            'duration': env.data.time,
            'reward_components': {k: v for k, v in info.items() if k.startswith('r_')}
        })
        
        print(f"Episode {episode_id + 1}: {steps} steps, {env.data.time:.1f}s, reward={total_reward:.2f}, success={success}")
    
    # Calculate metrics
    success_rate = np.mean([ep['success'] for ep in episode_results])
    avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
    avg_steps = np.mean([ep['steps'] for ep in episode_results])
    avg_duration = np.mean([ep['duration'] for ep in episode_results])
    
    print(f"\nResults:")
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Duration: {avg_duration:.1f}s")
    
    # Enhanced reward analysis
    if mode == 'research' and episode_results[0]['reward_components']:
        print("Enhanced Reward Components:")
        reward_components = {}
        for ep in episode_results:
            for comp_name, comp_value in ep['reward_components'].items():
                if comp_name not in reward_components:
                    reward_components[comp_name] = []
                reward_components[comp_name].append(comp_value)
        
        for comp, values in reward_components.items():
            avg_comp = np.mean(values)
            print(f"  {comp}: {avg_comp:.3f}")
    
    return {
        'model_name': model_name,
        'mode': mode,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_duration': avg_duration,
        'episodes': len(episode_results),
        'reward_components': reward_components if mode == 'research' else {}
    }


def main():
    """Main evaluation."""
    print("=== Proper BC Model Evaluation ===")
    
    # Model paths
    baseline_path = "checkpoints/checkpoints_baseline/torch_both_seed123_ep1_steps180000.pt"
    enhanced_path = "checkpoints/checkpoints_enhanced/torch_both_seed123_ep1_steps18000.pt"
    
    results = []
    
    # Evaluate baseline model
    if os.path.exists(baseline_path):
        baseline_fsm = evaluate_model_with_actual_model(baseline_path, "Baseline", episodes=5, mode='fsm')
        results.append(baseline_fsm)
        
        baseline_research = evaluate_model_with_actual_model(baseline_path, "Baseline", episodes=5, mode='research')
        results.append(baseline_research)
    else:
        print(f"Baseline model not found: {baseline_path}")
    
    # Evaluate enhanced model
    if os.path.exists(enhanced_path):
        enhanced_fsm = evaluate_model_with_actual_model(enhanced_path, "Enhanced", episodes=5, mode='fsm')
        results.append(enhanced_fsm)
        
        enhanced_research = evaluate_model_with_actual_model(enhanced_path, "Enhanced", episodes=5, mode='research')
        results.append(enhanced_research)
    else:
        print(f"Enhanced model not found: {enhanced_path}")
    
    # Print comparison summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Model':<12} {'Mode':<10} {'Success':<8} {'Reward':<8} {'Steps':<8} {'Duration':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['model_name']:<12} {result['mode']:<10} {result['success_rate']:<8.1%} {result['avg_reward']:<8.2f} {result['avg_steps']:<8.1f} {result['avg_duration']:<8.1f}")
    
    # Calculate improvements
    if len(results) >= 4:
        baseline_fsm_success = results[0]['success_rate']
        enhanced_fsm_success = results[2]['success_rate']
        
        baseline_fsm_reward = results[0]['avg_reward']
        enhanced_fsm_reward = results[2]['avg_reward']
        
        print(f"\n=== IMPROVEMENTS (FSM Mode) ===")
        print(f"Success Rate: {baseline_fsm_success:.1%} → {enhanced_fsm_success:.1%} ({enhanced_fsm_success - baseline_fsm_success:+.1%})")
        print(f"Average Reward: {baseline_fsm_reward:.2f} → {enhanced_fsm_reward:.2f} ({enhanced_fsm_reward - baseline_fsm_reward:+.2f})")
        
        if enhanced_fsm_success > baseline_fsm_success:
            print("✅ Enhanced model shows improved success rate!")
        if enhanced_fsm_reward > baseline_fsm_reward:
            print("✅ Enhanced model shows improved reward!")
    
    return results


if __name__ == "__main__":
    results = main()
