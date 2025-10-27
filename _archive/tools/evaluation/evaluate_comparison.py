#!/usr/bin/env python3
"""
Model Comparison Evaluation Script

Compare baseline vs enhanced BC models using the new evaluation suite.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from passive_walker.bc.evaluate import EvaluationConfig, evaluate_model_comprehensive
from passive_walker.bc.play import play_torch
from passive_walker.core.env import PassiveWalkerEnv


def evaluate_model_simple(checkpoint_path: str, model_name: str, episodes: int = 5):
    """Simple evaluation using existing play.py functionality."""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Use existing play functionality
    results = play_torch(
        ckpt_path=checkpoint_path,
        meta_path=checkpoint_path.replace('.pt', '_meta.json'),
        episodes=episodes,
        seconds=25.0,
        seed=42,
        headless=True
    )
    
    # Debug: print available keys
    print(f"Available keys: {list(results.keys())}")
    
    # Extract key metrics
    success_rate = (len(results["episodes"]) - results["falls"]) / max(1, len(results["episodes"]))
    
    # Calculate average reward and steps from episodes
    episodes = results["episodes"]
    if episodes:
        avg_reward = np.mean([ep.get("reward", 0.0) for ep in episodes])
        avg_steps = np.mean([ep.get("steps", 0) for ep in episodes])
    else:
        avg_reward = 0.0
        avg_steps = 0.0
    
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'model_name': model_name,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episodes': len(results["episodes"]),
        'falls': results["falls"]
    }


def evaluate_with_enhanced_rewards(checkpoint_path: str, model_name: str, episodes: int = 5):
    """Evaluate using research mode with enhanced rewards."""
    print(f"\n=== Evaluating {model_name} with Enhanced Rewards ===")
    
    # Create environment with research mode
    env = PassiveWalkerEnv(mode='research', ctrl_hz=100)
    
    # Load model (simplified - in practice you'd load the actual model)
    import torch
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Run episodes
    episode_results = []
    for episode_id in range(episodes):
        obs, _ = env.reset(seed=42 + episode_id)
        total_reward = 0.0
        steps = 0
        
        while env.data.time < 25.0 and steps < 2500:
            # Get random action for now (in practice you'd use the model)
            action = env.action_space.sample()
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
    
    # Calculate metrics
    success_rate = np.mean([ep['success'] for ep in episode_results])
    avg_reward = np.mean([ep['total_reward'] for ep in episode_results])
    avg_steps = np.mean([ep['steps'] for ep in episode_results])
    
    # Calculate enhanced reward metrics
    reward_components = {}
    for ep in episode_results:
        for comp_name, comp_value in ep['reward_components'].items():
            if comp_name not in reward_components:
                reward_components[comp_name] = []
            reward_components[comp_name].append(comp_value)
    
    avg_components = {comp: np.mean(values) for comp, values in reward_components.items()}
    
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print("Enhanced Reward Components:")
    for comp, value in avg_components.items():
        print(f"  {comp}: {value:.3f}")
    
    return {
        'model_name': f"{model_name}_enhanced",
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episodes': len(episode_results),
        'reward_components': avg_components
    }


def main():
    """Main comparison evaluation."""
    print("=== BC Model Comparison Evaluation ===")
    
    # Model paths
    hip_path = "experiments/models/torch_hip_seed123_ep1_steps9000.pt"
    both_path = "experiments/models/torch_both_seed456_ep1_steps9000.pt"
    
    results = []
    
    # Evaluate hip model
    if os.path.exists(hip_path):
        hip_results = evaluate_model_simple(hip_path, "Hip Control")
        results.append(hip_results)
    else:
        print(f"Hip model not found: {hip_path}")
    
    # Evaluate both joints model
    if os.path.exists(both_path):
        both_results = evaluate_model_simple(both_path, "Both Joints")
        results.append(both_results)
    else:
        print(f"Both joints model not found: {both_path}")
    
    # Print comparison summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Model':<20} {'Success Rate':<12} {'Avg Reward':<12} {'Avg Steps':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['success_rate']:<12.1%} {result['avg_reward']:<12.2f} {result['avg_steps']:<12.1f}")
    
    # Calculate improvements
    if len(results) >= 2:
        baseline_success = results[0]['success_rate']
        enhanced_success = results[2]['success_rate'] if len(results) > 2 else results[1]['success_rate']
        
        baseline_reward = results[0]['avg_reward']
        enhanced_reward = results[2]['avg_reward'] if len(results) > 2 else results[1]['avg_reward']
        
        print(f"\n=== IMPROVEMENTS ===")
        print(f"Success Rate: {baseline_success:.1%} → {enhanced_success:.1%} ({enhanced_success - baseline_success:+.1%})")
        print(f"Average Reward: {baseline_reward:.2f} → {enhanced_reward:.2f} ({enhanced_reward - baseline_reward:+.2f})")
        
        if enhanced_success > baseline_success:
            print("✅ Enhanced model shows improved success rate!")
        if enhanced_reward > baseline_reward:
            print("✅ Enhanced model shows improved reward!")
    
    return results


if __name__ == "__main__":
    results = main()
