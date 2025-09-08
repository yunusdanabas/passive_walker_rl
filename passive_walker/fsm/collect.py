"""
FSM Data Collection

Simple data collection for FSM walking behavior.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from pathlib import Path
from passive_walker.core.env import PassiveWalkerEnv


def main():
    """Collect FSM walking data."""
    parser = argparse.ArgumentParser("FSM data collection")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=512, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Disable GUI")
    parser.set_defaults(gui=False)
    args = parser.parse_args()

    # Create output directory if specified
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    # Initialize environment
    env = PassiveWalkerEnv(mode="fsm", use_gui=args.gui)
    
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Collecting {args.episodes} episodes of {args.steps} steps each...")
    
    total_steps = 0
    successful_episodes = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed)
        
        # Pre-allocate episode buffers
        obs_buffer = np.zeros((args.steps + 1, 11), dtype=np.float32)
        act_buffer = np.zeros((args.steps, 3), dtype=np.float32)  # FSM uses zeros
        rew_buffer = np.zeros(args.steps, dtype=np.float32)
        done_buffer = np.zeros(args.steps, dtype=bool)
        
        obs_buffer[0] = obs
        episode_steps = 0
        
        for step in range(args.steps):
            # FSM mode ignores actions
            obs, reward, done, info = env.step(np.zeros(3))
            
            # Store data
            obs_buffer[step + 1] = obs
            act_buffer[step] = np.zeros(3)  # FSM uses zero actions
            rew_buffer[step] = reward
            done_buffer[step] = done
            episode_steps += 1
            
            if done:
                break
        
        # Save episode data if output directory specified
        if args.out:
            episode_file = os.path.join(args.out, f"episode_{episode:06d}.npz")
            np.savez_compressed(
                episode_file,
                obs=obs_buffer[:episode_steps + 1],
                act=act_buffer[:episode_steps],
                rew=rew_buffer[:episode_steps],
                done=done_buffer[:episode_steps]
            )
            print(f"Saved episode {episode} to {episode_file}")
        
        total_steps += episode_steps
        if not done:  # Episode completed successfully
            successful_episodes += 1
        
        print(f"Episode {episode + 1}/{args.episodes}: {episode_steps} steps, "
              f"pitch={info.get('pitch_abs', 0):.3f}, fell={info.get('fell', False)}")

    # Print summary
    print(f"\nCollection complete:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Successful: {successful_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Success rate: {successful_episodes/args.episodes*100:.1f}%")
    
    env.close()


if __name__ == "__main__":
    main()