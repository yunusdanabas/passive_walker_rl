#!/usr/bin/env python3
"""
Play PPO Model in MuJoCo

Loads a trained PPO model and runs it in the environment with visualization.
"""

import argparse
import torch
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.ppo.models import create_actor_critic


def load_ppo_model(model_path, config=None):
    """Load a trained PPO model."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model info from checkpoint or config
    if 'config' in checkpoint:
        config_info = checkpoint['config']
        model_type = config_info.get('model_type', 'mlp')
        hidden_sizes = config_info.get('hidden_sizes', [64, 64])
        hidden_size = config_info.get('hidden_size', 64)
        num_layers = config_info.get('num_layers', 1)
    else:
        # Default to MLP if no config
        model_type = 'mlp'
        hidden_sizes = [64, 64]
        hidden_size = 64
        num_layers = 1
    
    print(f"Model type: {model_type}")
    
    # Create model architecture
    if model_type == 'mlp':
        model = create_actor_critic('mlp', obs_dim=17, action_dim=3, hidden_sizes=hidden_sizes)
    else:
        model = create_actor_critic(model_type, obs_dim=17, action_dim=3, 
                                    hidden_size=hidden_size, num_layers=num_layers)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights directly")
    
    model.eval()
    print(f"Model loaded successfully!\n")
    
    return model, model_type


def play_model(model, model_type, n_episodes=3, headless=False, deterministic=True):
    """Run the model in the environment."""
    
    print(f"{'='*70}")
    print(f"RUNNING MODEL IN MUJOCO")
    print(f"{'='*70}\n")
    print(f"Episodes: {n_episodes}")
    print(f"Headless: {headless}")
    print(f"Deterministic: {deterministic}")
    print()
    
    # Create environment
    env = PassiveWalkerEnv(mode='research', use_gui=not headless)
    
    episode_results = []
    
    for episode in range(n_episodes):
        print(f"\n{'─'*70}")
        print(f"EPISODE {episode + 1}/{n_episodes}")
        print(f"{'─'*70}")
        
        # Reset environment
        obs, _ = env.reset()
        
        episode_return = 0.0
        episode_length = 0
        done = False
        
        if model_type in ['lstm', 'gru']:
            # Reset hidden state for temporal models
            if hasattr(model, 'reset_hidden'):
                hidden_state = model.reset_hidden(1)  # batch_size=1
            else:
                hidden_state = None
        
        print("Starting episode...")
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                if model_type in ['lstm', 'gru']:
                    # Temporal model
                    if hidden_state is not None:
                        action_output = model.get_action(obs_tensor, hidden_state, deterministic=deterministic)
                        action, log_prob, value, hidden_state = action_output
                    else:
                        action_output = model.get_action(obs_tensor, deterministic=deterministic)
                        if len(action_output) == 4:
                            action, log_prob, value, _ = action_output
                        else:
                            action, log_prob, value = action_output
                else:
                    # MLP model
                    action_output = model.get_action(obs_tensor, deterministic=deterministic)
                    if len(action_output) == 4:
                        action, log_prob, value, _ = action_output
                    else:
                        action, log_prob, value = action_output
                
                action = action.squeeze(0).cpu().numpy()
                
                # Add comprehensive action validation
                assert not np.isnan(action).any(), f"NaN in action: {action}"
                assert not np.isinf(action).any(), f"Inf in action: {action}"
                assert (np.abs(action) < 10).all(), f"Action out of range: {action}"
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            episode_return += reward
            episode_length += 1
            obs = next_obs
            
            # Render environment (for GUI visualization)
            env.render()
            
            # Print progress every 500 steps
            if episode_length % 500 == 0:
                print(f"  Step {episode_length}: Return = {episode_return:.2f}, "
                      f"Distance = {info.get('dx', 0):.2f}")
        
        # Episode complete
        print(f"\nEpisode {episode + 1} complete!")
        print(f"  Total Steps: {episode_length}")
        print(f"  Total Return: {episode_return:.2f}")
        print(f"  Episode Length: {episode_length} steps")
        print(f"  Final Time: {info.get('time', 0):.2f}s")
        print(f"  Status: {'✅ Success' if episode_length > 100 else '❌ Failure'}")
        
        episode_results.append({
            'length': episode_length,
            'return': episode_return,
            'success': episode_length > 100
        })
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nEpisodes completed: {len(episode_results)}")
    print(f"Success rate: {sum(r['success'] for r in episode_results) / len(episode_results):.1%}")
    print(f"Average return: {np.mean([r['return'] for r in episode_results]):.2f}")
    print(f"Average length: {np.mean([r['length'] for r in episode_results]):.1f} steps")
    print(f"Max length: {max([r['length'] for r in episode_results])} steps")
    
    success_rate = sum(r['success'] for r in episode_results) / len(episode_results)
    if success_rate == 1.0:
        print(f"\n✅ MODEL WORKS PERFECTLY!")
    elif success_rate >= 0.8:
        print(f"\n✅ MODEL WORKS WELL")
    else:
        print(f"\n❌ MODEL HAS ISSUES")
    
    print(f"{'='*70}\n")
    
    return episode_results


def main():
    parser = argparse.ArgumentParser(description="Play PPO model in MuJoCo")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.pth file)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of episodes to run")
    parser.add_argument("--headless", action="store_true",
                       help="Run without GUI (for servers)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return 1
    
    # Load model
    try:
        model, model_type = load_ppo_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run model
    try:
        play_model(model, model_type, args.episodes, args.headless, args.deterministic)
    except Exception as e:
        print(f"Error running model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
