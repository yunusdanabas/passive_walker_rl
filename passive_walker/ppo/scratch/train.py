#!/usr/bin/env python
"""
Train a PPO agent from scratch for the passive walker.

This script initializes a policy and critic from scratch (no BC pre-training),
then trains them using Proximal Policy Optimization (PPO) algorithm.
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from passive_walker.ppo.scratch import (
    XML_PATH, DATA_PPO_SCRATCH, set_device,
    save_policy_and_critic, save_model,
    DEFAULT_ITERATIONS, DEFAULT_ROLLOUT_STEPS,
    DEFAULT_PPO_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_GAMMA, DEFAULT_LAMBDA, DEFAULT_CLIP_EPS,
    DEFAULT_SIGMA, DEFAULT_POLICY_LR, DEFAULT_CRITIC_LR,
    DEFAULT_HZ
)
from passive_walker.ppo.scratch.utils import (
    initialize_policy,
    collect_trajectories,
    compute_advantages,
    policy_log_prob,
    plot_training_rewards
)
from passive_walker.envs.mujoco_env import PassiveWalkerEnv

# Define critic network
class Critic(eqx.Module):
    """Value function estimator network."""
    layers: list
    
    def __init__(self, in_size, hidden=512, key=None):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_size, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, 1, key=keys[2]),
        ]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
        return x.squeeze()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PPO from scratch")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                       help=f"Number of PPO iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS,
                       help=f"Steps to collect per iteration (default: {DEFAULT_ROLLOUT_STEPS})")
    parser.add_argument("--ppo-epochs", type=int, default=DEFAULT_PPO_EPOCHS,
                       help=f"Epochs per PPO update (default: {DEFAULT_PPO_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Minibatch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                       help=f"Discount factor (default: {DEFAULT_GAMMA})")
    parser.add_argument("--lam", type=float, default=DEFAULT_LAMBDA,
                       help=f"GAE lambda (default: {DEFAULT_LAMBDA})")
    parser.add_argument("--clip-eps", type=float, default=DEFAULT_CLIP_EPS,
                       help=f"PPO clipping epsilon (default: {DEFAULT_CLIP_EPS})")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                       help=f"Policy std dev (default: {DEFAULT_SIGMA})")
    parser.add_argument("--policy-lr", type=float, default=DEFAULT_POLICY_LR,
                       help=f"Policy learning rate (default: {DEFAULT_POLICY_LR})")
    parser.add_argument("--critic-lr", type=float, default=DEFAULT_CRITIC_LR,
                       help=f"Critic learning rate (default: {DEFAULT_CRITIC_LR})")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration")
    parser.add_argument("--hz", type=int, default=DEFAULT_HZ,
                       help=f"Simulation frequency (Hz) (default: {DEFAULT_HZ})")
    args = parser.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Init env and policy/critic from scratch
    dummy_env = PassiveWalkerEnv(xml_path=str(XML_PATH), simend=args.rollout_steps/args.hz, use_gui=False)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    
    env, get_scaled_action, get_env_action, policy = initialize_policy(
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        xml_path=str(XML_PATH), 
        simend=args.rollout_steps/args.hz, 
        sigma=args.sigma, 
        use_gui=False
    )
    
    # Initialize critic
    key = jax.random.PRNGKey(0)
    critic = Critic(obs_dim, hidden=512, key=key)
    
    # Initialize optimizers
    policy_opt = optax.adam(args.policy_lr)
    critic_opt = optax.adam(args.critic_lr)
    policy_state = policy_opt.init(eqx.filter(policy, eqx.is_array))
    critic_state = critic_opt.init(eqx.filter(critic, eqx.is_array))

    # Training loop
    reward_history = []

    for it in range(1, args.iterations + 1):
        # 1. Collect trajectory data
        traj = collect_trajectories(
            env=env,
            env_action_fn=get_env_action,
            scaled_action_fn=get_scaled_action,
            num_steps=args.rollout_steps,
            render=False
        )
        
        obs = jnp.array(traj["obs"], dtype=jnp.float32)
        acts = jnp.array(traj["scaled_actions"], dtype=jnp.float32)
        rewards = np.array(traj["rewards"])
        dones = np.array(traj["dones"])

        # 2. Compute GAE advantages/returns
        vals = np.array(jax.vmap(critic)(obs))
        adv, ret = compute_advantages(rewards, dones, vals, gamma=args.gamma, lam=args.lam)
        adv_j = jnp.array(adv, dtype=jnp.float32)
        ret_j = jnp.array(ret, dtype=jnp.float32)
        
        # 3. Save old policy log probabilities
        old_lp = policy_log_prob(policy, obs, acts, args.sigma)

        # 4. PPO update loop
        idxs = np.arange(obs.shape[0])
        for _ in range(args.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), args.batch_size):
                b = idxs[start:start+args.batch_size]
                # PPO policy update
                def loss_fn(policy_, obs_, acts_, old_lp_, adv_):
                    new_lp = policy_log_prob(policy_, obs_, acts_, args.sigma)
                    ratio = jnp.exp(new_lp - old_lp_)
                    clipped_obj = jnp.clip(ratio, 1-args.clip_eps, 1+args.clip_eps) * adv_
                    ppo_obj = -jnp.mean(jnp.minimum(ratio * adv_, clipped_obj))
                    return ppo_obj

                grads = jax.grad(loss_fn)(
                    policy, obs[b], acts[b], old_lp[b], adv_j[b])
                updates, policy_state = policy_opt.update(grads, policy_state)
                policy = eqx.apply_updates(policy, updates)
                
                # Critic update
                def vf_loss_fn(critic_params, critic_static, obs_, ret_):
                    critic_ = eqx.combine(critic_params, critic_static)
                    pred = jax.vmap(critic_)(obs_)
                    return jnp.mean((pred - ret_)**2)
                
                # Split critic into parameters and static components
                critic_params, critic_static = eqx.partition(critic, eqx.is_array)
                
                # Compute gradients with respect to parameters only
                vf_grads = jax.grad(vf_loss_fn)(critic_params, critic_static, obs[b], ret_j[b])
                
                # Update parameters
                vf_updates, critic_state = critic_opt.update(vf_grads, critic_state)
                critic_params = optax.apply_updates(critic_params, vf_updates)
                
                # Recombine parameters with static components
                critic = eqx.combine(critic_params, critic_static)

        # Print progress and save to history
        avg_reward = rewards.mean()
        reward_history.append(avg_reward)
        print(f"[PPO scratch] iter {it}/{args.iterations}  avg_rew={avg_reward:.2f}")

    # Save final model/critic/log
    save_policy_and_critic(policy, critic, DATA_PPO_SCRATCH, args.hz)
    
    # Save training log
    log = {"rewards": reward_history}
    log_path = DATA_PPO_SCRATCH / f"ppo_training_log_{args.hz}hz.pkl"
    save_model(log, log_path)
    print(f"[PPO scratch] saved log → {log_path}")
    
    # Plot training curve
    plot_training_rewards(reward_history, 
                         save_path=DATA_PPO_SCRATCH / f"ppo_training_curve_{args.hz}hz.png")

if __name__ == "__main__":
    main()
