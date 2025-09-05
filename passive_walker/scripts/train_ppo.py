"""
PPO training script.

Trains a PPO agent on the passive walker environment.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.rollout_buffer import MultiEnvRolloutBuffer


class PPOPolicy(nn.Module):
    """PPO policy network with separate actor and critic."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Actor head (outputs mean and log_std)
        self.actor_mean = nn.Linear(hidden_dims[1], act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dims[1], 1)

    def forward(self, obs):
        features = self.feature_extractor(obs)

        # Actor
        mean = torch.tanh(self.actor_mean(features))  # Actions in [-1, 1]
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)

        # Critic
        value = self.critic(features)

        return mean, std, value

    def get_action(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)

        if deterministic:
            action = mean
        else:
            dist = Normal(mean, std)
            action = dist.sample()

        return action, value

    def evaluate_actions(self, obs, actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_advantage = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage

    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy,
    optimizer,
    obs,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
):
    """Perform PPO update."""
    policy.train()

    for _ in range(epochs):
        # Forward pass
        log_probs, values, entropy = policy.evaluate_actions(obs, actions)

        # Compute ratios
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = actor_loss + value_coef * value_loss + entropy_coef * entropy_loss

        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return actor_loss.item(), value_loss.item(), entropy_loss.item()


def collect_rollouts(envs, policy, buffer, num_steps, device):
    """Collect rollouts from multiple environments."""
    policy.eval()

    # Reset environments
    for i in range(len(envs)):
        envs[i].reset()
        buffer.reset_env(i)

    # Collect data
    step = 0
    while step < num_steps:
        # Get actions from policy
        obs_list = []
        actions_list = []
        values_list = []

        for i, env in enumerate(envs):
            if not buffer.is_full(i):
                obs = env._get_obs()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, value = policy.get_action(obs_tensor)

                obs_list.append(obs)
                actions_list.append(action.cpu().numpy().squeeze())
                values_list.append(value.cpu().numpy().squeeze())

        # Step environments
        for i, env in enumerate(envs):
            if not buffer.is_full(i):
                action = actions_list[i]
                obs, reward, done, info = env.step(action)

                # Add to buffer
                buffer.add(i, obs_list[i], action, reward, done, info)

                if done:
                    env.reset()

        step += 1


def train_ppo(
    cfg,
    num_envs=4,
    total_steps=100000,
    rollout_len=2048,
    lr=3e-4,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    bc_init_path=None,
    device="cpu",
    seed=42,
):
    """Train PPO agent."""

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environments
    envs = []
    for _ in range(num_envs):
        env = PassiveWalkerEnv(cfg, use_gui=False)
        envs.append(env)

    # Create buffer
    buffer = MultiEnvRolloutBuffer(
        num_envs=num_envs,
        rollout_len=rollout_len,
        obs_dim=11,  # From env observation space
        act_dim=3,  # From env action space
        store_extras=True,
    )

    # Create policy
    policy = PPOPolicy(obs_dim=11, act_dim=3).to(device)

    # Load BC initialization if provided
    if bc_init_path and os.path.exists(bc_init_path):
        print(f"Loading BC initialization from {bc_init_path}")
        bc_data = torch.load(bc_init_path, map_location=device)
        policy.load_state_dict(bc_data["model_state_dict"])
        print("BC initialization loaded successfully")

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    step = 0
    episode = 0
    episode_rewards = []
    training_stats = []

    print(f"Starting PPO training for {total_steps} steps...")
    print(f"Environments: {num_envs}, Rollout length: {rollout_len}")

    while step < total_steps:
        # Collect rollouts
        collect_rollouts(envs, policy, buffer, rollout_len, device)

        # Get collected data
        data = buffer.stacked()
        if not data:
            continue

        obs = torch.FloatTensor(data["obs"]).to(device)
        actions = torch.FloatTensor(data["act"]).to(device)
        rewards = data["rew"]
        dones = data["done"]

        # Flatten for training
        obs = obs.view(-1, obs.shape[-1])
        actions = actions.view(-1, actions.shape[-1])
        rewards = rewards.flatten()
        dones = dones.flatten()

        # Get old log probs
        with torch.no_grad():
            old_log_probs, values, _ = policy.evaluate_actions(obs, actions)
            values = values.squeeze().cpu().numpy()

        # Compute GAE
        advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
        advantages = torch.FloatTensor(advantages).to(device)
        returns = torch.FloatTensor(returns).to(device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        actor_loss, value_loss, entropy_loss = ppo_update(
            policy,
            optimizer,
            obs,
            actions,
            old_log_probs,
            advantages,
            returns,
            clip_ratio,
            value_coef,
            entropy_coef,
            epochs,
        )

        # Logging
        episode_reward = rewards.sum()
        episode_rewards.append(episode_reward)
        episode += 1

        training_stats.append(
            {
                "episode": episode,
                "step": step,
                "episode_reward": episode_reward,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
            }
        )

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"Episode {episode:4d}: Step {step:6d}, "
                f"Avg Reward: {avg_reward:6.2f}, "
                f"Actor Loss: {actor_loss:.4f}, "
                f"Value Loss: {value_loss:.4f}"
            )

        step += len(rewards)

    # Close environments
    for env in envs:
        env.close()

    return policy, training_stats


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent")
    parser.add_argument(
        "--config",
        type=str,
        default="passive_walker/configs/ppo_train.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/ppo", help="Output directory for trained policy"
    )
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--rollout_len", type=int, default=2048, help="Rollout length per update")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy loss coefficient")
    parser.add_argument("--epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument(
        "--bc_init", type=str, default=None, help="Path to BC policy for initialization"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load config
    cfg = load_walker_config(args.config)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Train policy
    policy, stats = train_ppo(
        cfg=cfg,
        num_envs=args.num_envs,
        total_steps=args.total_steps,
        rollout_len=args.rollout_len,
        lr=args.lr,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs,
        bc_init_path=args.bc_init,
        device=args.device,
        seed=args.seed,
    )

    # Save policy and stats
    model_path = os.path.join(args.out_dir, "policy.pt")
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": 11,
            "act_dim": 3,
            "training_stats": stats,
        },
        model_path,
    )

    # Save training summary
    summary = {
        "config": args.config,
        "total_steps": args.total_steps,
        "num_envs": args.num_envs,
        "rollout_len": args.rollout_len,
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "clip_ratio": args.clip_ratio,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "epochs": args.epochs,
        "bc_init": args.bc_init,
        "device": args.device,
        "seed": args.seed,
        "final_episode_reward": stats[-1]["episode_reward"] if stats else 0,
        "avg_reward_last_10": (
            np.mean([s["episode_reward"] for s in stats[-10:]]) if len(stats) >= 10 else 0
        ),
    }

    summary_path = os.path.join(args.out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete!")
    print(f"Policy saved to: {model_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Final episode reward: {stats[-1]['episode_reward']:.2f}")


if __name__ == "__main__":
    main()
