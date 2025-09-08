"""
PPO Training

Simple PPO training for passive walker reinforcement learning.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from passive_walker.core.env import PassiveWalkerEnv


# =====================
# PPO Parameters
# =====================
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPSILON = 0.2
PPO_EPOCHS = 4
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
UPDATE_FREQ = 2048
EVAL_FREQ = 10


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, obs_dim=11, action_dim=3, hidden_size=64):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, obs):
        features = self.feature_extractor(obs)
        action = self.actor(features)
        value = self.critic(features)
        return action, value
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy."""
        with torch.no_grad():
            action, value = self.forward(obs)
            if not deterministic:
                # Add noise for exploration (simplified)
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1, 1)
        return action, value


def compute_gae(rewards, values, dones, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns, 
               ppo_epochs=PPO_EPOCHS, epsilon=PPO_EPSILON):
    """Perform PPO update."""
    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    old_log_probs = torch.FloatTensor(old_log_probs)
    advantages = torch.FloatTensor(advantages)
    returns = torch.FloatTensor(returns)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    for _ in range(ppo_epochs):
        # Forward pass
        new_actions, values = model(states)
        
        # Compute policy loss (simplified - no log probs for continuous actions)
        action_diff = (new_actions - actions).pow(2).sum(dim=1)
        policy_loss = -advantages * action_diff.mean()
        
        # Value loss
        value_loss = (values.squeeze() - returns).pow(2).mean()
        
        # Total loss
        total_loss = policy_loss + VALUE_LOSS_COEF * value_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()


def collect_rollout(env, model, steps):
    """Collect rollout data."""
    states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
    
    obs, _ = env.reset()
    
    for _ in range(steps):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action, value = model.get_action(obs_tensor)
        action = action.squeeze(0).numpy()
        value = value.squeeze(0).item()
        
        next_obs, reward, done, info = env.step(action)
        
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        log_probs.append(0.0)  # Simplified - no log probs
        
        obs = next_obs
        
        if done:
            obs, _ = env.reset()
    
    return states, actions, rewards, dones, values, log_probs


def evaluate_policy(env, model, num_episodes=5):
    """Evaluate policy performance."""
    total_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 1000:  # Max episode length
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, _ = model.get_action(obs_tensor, deterministic=True)
            action = action.squeeze(0).numpy()
            
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser("PPO training")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create output directory
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    
    # Create environment and model
    env = PassiveWalkerEnv(mode="research", use_gui=False)
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Training for {args.timesteps} timesteps")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    total_steps = 0
    update_count = 0
    
    while total_steps < args.timesteps:
        # Collect rollout
        states, actions, rewards, dones, values, log_probs = collect_rollout(
            env, model, UPDATE_FREQ
        )
        
        # Compute advantages and returns
        advantages, returns = compute_gae(rewards, values, dones)
        
        # PPO update
        ppo_update(model, optimizer, states, actions, log_probs, advantages, returns)
        
        total_steps += len(states)
        update_count += 1
        
        # Logging
        mean_reward = np.mean(rewards)
        print(f"Update {update_count}: steps={total_steps}, mean_reward={mean_reward:.3f}")
        
        # Evaluation
        if update_count % EVAL_FREQ == 0:
            eval_results = evaluate_policy(env, model)
            print(f"Eval: reward={eval_results['mean_reward']:.3f}±{eval_results['std_reward']:.3f}, "
                  f"length={eval_results['mean_length']:.1f}±{eval_results['std_length']:.1f}")
            
            # Save model
            if args.out:
                torch.save(model.state_dict(), os.path.join(args.out, f"model_{update_count}.pth"))
    
    print("Training complete!")
    
    # Save final model
    if args.out:
        torch.save(model.state_dict(), os.path.join(args.out, "final_model.pth"))
        print(f"Final model saved to {args.out}")
    
    env.close()


if __name__ == "__main__":
    main()