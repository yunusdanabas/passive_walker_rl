"""
FSM data collection script using rollout buffer.

Collects expert trajectories from FSM-controlled walker and saves them for BC training.
"""

import argparse
import os
import json
import numpy as np
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.rollout_buffer import RolloutBuffer


def collect_episode(env: PassiveWalkerEnv, buffer: RolloutBuffer, episode_id: int) -> dict:
    """Collect one episode and return performance metrics."""
    obs, _ = env.reset()
    buffer.reset()

    episode_metrics = {
        "episode_id": episode_id,
        "total_reward": 0.0,
        "steps": 0,
        "fell": False,
        "final_x": 0.0,
    }

    done = False
    while not buffer.is_full() and not done:
        # FSM mode uses zero actions (FSM overrides them)
        act = np.zeros(3, dtype=np.float32)
        obs2, rew, done, info = env.step(act)

        # Extract reward components for analysis
        extras = {k: info[k] for k in info.keys() if k.startswith("r_")}

        # Store experience in buffer
        buffer.add(obs, act, rew, done, info, extras)

        # Update episode metrics
        episode_metrics["total_reward"] += rew
        episode_metrics["steps"] += 1
        episode_metrics["fell"] = info.get("fell", False)
        episode_metrics["final_x"] = info.get("dx", 0.0)  # Forward progress

        obs = obs2

    return episode_metrics


def main():
    """Main function for FSM data collection."""
    parser = argparse.ArgumentParser(description="Collect FSM data for BC training")
    parser.add_argument(
        "--config",
        type=str,
        default="passive_walker/configs/walker.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100, help="Number of episodes to collect"
    )
    parser.add_argument("--rollout_len", type=int, default=4096, help="Maximum steps per episode")
    parser.add_argument(
        "--output_dir", type=str, default="data/bc/raw", help="Output directory for collected data"
    )
    parser.add_argument(
        "--min_quality",
        type=float,
        default=0.0,
        help="Minimum episode quality to save (total reward threshold)",
    )
    parser.add_argument("--use_gui", action="store_true", help="Use GUI for visualization")

    args = parser.parse_args()

    # Load configuration and force FSM mode
    cfg = load_walker_config(args.config)
    cfg.mode = "fsm"  # Force FSM mode for data collection

    # Initialize environment and output directory
    env = PassiveWalkerEnv(cfg, use_gui=args.use_gui)
    os.makedirs(args.output_dir, exist_ok=True)

    # Create rollout buffer for experience storage
    buffer = RolloutBuffer(
        rollout_len=args.rollout_len,
        obs_dim=11,  # Environment observation space
        act_dim=3,  # Environment action space
        store_extras=True,
    )

    # Initialize collection statistics
    collected_episodes = 0
    total_episodes = 0
    all_metrics = []
    rejected_count = 0

    print(f"Collecting {args.num_episodes} episodes...")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum quality threshold: {args.min_quality}")
    print("Progress: [", end="", flush=True)

    try:
        while collected_episodes < args.num_episodes:
            total_episodes += 1

            # Collect episode
            metrics = collect_episode(env, buffer, total_episodes)
            all_metrics.append(metrics)

            # Check quality threshold
            if metrics["total_reward"] >= args.min_quality:
                # Save episode
                episode_file = os.path.join(
                    args.output_dir, f"episode_{collected_episodes:06d}.npz"
                )
                buffer.save_npz(episode_file)

                collected_episodes += 1
                print("✓", end="", flush=True)  # Show progress
            else:
                rejected_count += 1
                print(".", end="", flush=True)  # Show rejected episodes as dots

            # Print progress every 10 episodes
            if total_episodes % 10 == 0:
                print(
                    f"] {collected_episodes}/{args.num_episodes} collected, {rejected_count} rejected",
                    flush=True,
                )
                print("Progress: [", end="", flush=True)

    except KeyboardInterrupt:
        print("\nCollection interrupted by user")

    finally:
        env.close()

    # Print final progress line
    print(f"] {collected_episodes}/{args.num_episodes} collected, {rejected_count} rejected")

    # Save collection summary
    summary_file = os.path.join(args.output_dir, "collection_summary.json")
    summary = {
        "total_episodes_attempted": total_episodes,
        "episodes_collected": collected_episodes,
        "collection_rate": collected_episodes / max(1, total_episodes),
        "min_quality_threshold": args.min_quality,
        "rollout_len": args.rollout_len,
        "episode_metrics": all_metrics,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Collection complete!")
    print(
        f"📊 Episodes collected: {collected_episodes}/{total_episodes} ({collected_episodes / max(1, total_episodes):.1%})"
    )
    print(f"📁 Data saved to: {args.output_dir}")
    print(f"📋 Summary: {summary_file}")


if __name__ == "__main__":
    main()
