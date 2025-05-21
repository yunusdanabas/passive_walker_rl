"""
Utility functions for PPO implementation from scratch.

This module provides common functionality used across the PPO pipeline:
- Serialization (saving/loading pickle files)
- Policy initialization from scratch
- On-policy trajectory collection
- Advantage estimation (GAE)
- Policy log probability calculations
- Visualization utilities
"""

import pickle
import os
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from pathlib import Path
from passive_walker.envs.mujoco_env import PassiveWalkerEnv
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.ppo.scratch import DATA_PPO_SCRATCH

# — serialization —

def save_pickle(obj, path):
    """
    Save an object to a pickle file.
    
    Args:
        obj: Python object to serialize
        path: Path to save the pickle file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Load an object from a pickle file.
    
    Args:
        path: Path to the pickle file
        
    Returns:
        The deserialized object
    """
    with open(path, "rb") as f:
        return pickle.load(f)

# — policy initialization —

def initialize_policy(
    obs_dim: int = None,
    act_dim: int = None,
    xml_path: str = None,
    simend: float = None,
    sigma: float = 0.1,
    use_gui: bool = False
):
    """
    Initialize a fresh policy and the environment.
    
    Args:
        obs_dim: Observation dimension (optional, will be inferred from env if None)
        act_dim: Action dimension (optional, will be inferred from env if None)
        xml_path: Path to MuJoCo XML model file
        simend: Simulation end time
        sigma: Standard deviation for exploration
        use_gui: Whether to use GUI rendering
        
    Returns:
        Tuple of (env, get_scaled_action, get_env_action, policy_model)
    """
    # First create environment to get observation and action spaces
    env = PassiveWalkerEnv(
        xml_path=xml_path,
        simend=simend,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=use_gui,
    )
    
    # Infer dimensions from environment if not provided
    if obs_dim is None:
        obs_dim = env.observation_space.shape[0]
    if act_dim is None:
        act_dim = env.action_space.shape[0]
    
    key = jax.random.PRNGKey(0)
    policy_model = HipKneeController(input_size=obs_dim, hidden_size=128, key=key)

    def get_scaled(obs_jnp: jnp.ndarray) -> jnp.ndarray:
        """Get scaled actions from policy."""
        return policy_model(obs_jnp)

    def get_env(obs_jnp: jnp.ndarray) -> np.ndarray:
        """Get environment actions from policy."""
        return np.array(get_scaled(obs_jnp), dtype=np.float32)

    return env, get_scaled, get_env, policy_model

# — on‐policy rollout buffer —

def collect_trajectories(
    env,
    env_action_fn,
    scaled_action_fn=None,
    num_steps: int = 1024,
    render: bool = False
):
    """
    Collect on‐policy rollouts from the environment.
    
    Args:
        env: Environment instance
        env_action_fn: Function that maps observations to environment actions
        scaled_action_fn: Function that maps observations to scaled actions (optional)
        num_steps: Number of steps to collect
        render: Whether to render the environment
        
    Returns:
        Dictionary containing trajectory data with keys:
        - obs: Observations
        - actions: Actions taken in environment
        - scaled_actions: Scaled versions of actions
        - rewards: Rewards received
        - next_obs: Next observations
        - dones: Terminal flags
    """
    o = env.reset()
    buf = {k: [] for k in ("obs","actions","scaled_actions","rewards","next_obs","dones")}

    for _ in range(num_steps):
        o_jnp = jnp.array(o, dtype=jnp.float32)
        a     = env_action_fn(o_jnp)
        sa    = scaled_action_fn(o_jnp) if scaled_action_fn else a

        o2, r, done, _ = env.step(np.array(a, dtype=np.float32))
        buf["obs"].append(o)
        buf["actions"].append(a)
        buf["scaled_actions"].append(np.array(sa, dtype=np.float32))
        buf["rewards"].append(r)
        buf["next_obs"].append(o2)
        buf["dones"].append(done)

        if render:
            env.render()
        o = o2 if not done else env.reset()

    # stack into numpy arrays
    return {k: np.array(v) for k, v in buf.items()}

# — GAE —

def compute_advantages(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    gamma: float = 0.99,
    lam:   float = 0.95
):
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Array of rewards
        dones: Array of terminal flags
        values: Array of value estimates
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        nextval = values[t+1] if t+1 < T else values[t]
        delta = rewards[t] + gamma * nextval * nonterm - values[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam
    returns = adv + values
    return adv, returns

# — Gaussian log‐prob helper —

def policy_log_prob(
    policy_model,
    obs_jnp: jnp.ndarray,
    acts_jnp: jnp.ndarray,
    sigma: float
) -> jnp.ndarray:
    """
    Compute log probability of actions under a Gaussian policy.
    
    Args:
        policy_model: Policy model that outputs action means
        obs_jnp: JAX array of observations
        acts_jnp: JAX array of actions
        sigma: Standard deviation of the Gaussian policy
        
    Returns:
        JAX array of log probabilities
    """
    mean    = jax.vmap(policy_model)(obs_jnp)
    var     = sigma**2
    log_std = jnp.log(sigma)
    lp = -0.5 * (((acts_jnp - mean)**2)/var + 2*log_std + jnp.log(2*jnp.pi))
    return jnp.sum(lp, axis=-1)

# — Plotting utilities —

def plot_training_rewards(rewards, save_path=DATA_PPO_SCRATCH / "ppo_training_curve.png", title="Average Reward per PPO Iteration", print_stats=True):
    """
    Plot average reward per PPO iteration.
    
    Args:
        rewards: List or array of average rewards per iteration
        save_path: Path to save the figure (default: DATA_PPO_SCRATCH/ppo_training_curve.png)
        title: Title for the plot
        print_stats: Whether to print statistics about the rewards
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title(title)
    plt.grid(True)
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved training curve → {save_path}")
    
    plt.show()
    
    # Print statistics if requested
    if print_stats and len(rewards) > 0:
        print(f"Number of training iterations: {len(rewards)}")
        print(f"Final average reward: {rewards[-1]:.2f}")
        
        # Print improvement statistics if we have more than one iteration
        if len(rewards) > 1:
            improvement = rewards[-1] - rewards[0]
            percent_improvement = (improvement / abs(rewards[0] + 1e-10)) * 100  # Avoid division by zero
            print(f"Improvement: {improvement:.2f} ({percent_improvement:.1f}%)")

def analyze_training_log(log_path=DATA_PPO_SCRATCH / "ppo_training_log.pkl", save_path=DATA_PPO_SCRATCH / "ppo_training_curve.png", title="Average Reward per PPO Iteration"):
    """
    Load and analyze a training log, plotting the rewards.
    
    Args:
        log_path: Path to the training log pickle file (default: DATA_PPO_SCRATCH/ppo_training_log.pkl)
        save_path: Path to save the plot (default: DATA_PPO_SCRATCH/ppo_training_curve.png) 
        title: Title for the plot
        
    Returns:
        None
    """
    log = load_pickle(log_path)
    rewards = log["rewards"]
    plot_training_rewards(rewards, save_path=save_path, title=title)

def plot_joint_and_reward(traj_obs, rewards, save_prefix=DATA_PPO_SCRATCH / "ppo_eval"):
    """
    Plot joint positions and rewards over time.
    
    Args:
        traj_obs: Array of trajectory observations
        rewards: Array of rewards
        save_prefix: Prefix for saving plot files
    """
    t = np.arange(len(rewards))
    n_q = traj_obs.shape[1] // 2 if traj_obs.shape[1] > 6 else traj_obs.shape[1] # safety for qpos
    plt.figure(figsize=(8,4))
    for i in range(min(3, n_q)):
        plt.plot(t, traj_obs[:, i], label=f"Joint {i+1}")
    plt.legend()
    plt.title("Joint Positions Over Time")
    plt.xlabel("Time step")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_joint_positions.png", dpi=150)
    plt.show()

    plt.figure(figsize=(8,3))
    plt.plot(t, rewards, color='tab:blue')
    plt.title("Reward per Step")
    plt.xlabel("Time step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_rewards.png", dpi=150)
    plt.show()