"""
Tiny PPO sanity check for the Brax passive-walker.

This module implements a minimal PPO (Proximal Policy Optimization) training setup
for a passive walker environment using Brax and MuJoCo. It includes both training
and visualization capabilities.

Usage
-----
Train only (no GUI):
    python -m passive_walker.brax.tiny_ppo_sanity

Train and replay in MuJoCo:
    python -m passive_walker.brax.tiny_ppo_sanity --mujoco 10
    where the argument to `--mujoco` is the GUI episode length in seconds.
"""

from __future__ import annotations
import argparse
import pickle
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import mujoco
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax import linen as nn
import optax
import brax
from brax.io import mjcf
from mujoco.glfw import glfw

from passive_walker.brax import RESULTS_BRAX, XML_PATH
from passive_walker.envs.brax_env import BraxPassiveWalker
from passive_walker.envs.mujoco_env import PassiveWalkerEnv
from passive_walker.brax.utils import uint64_patch

from brax.training.agents.ppo import train as ppo_train     
from brax.training.agents.ppo import networks as ppo_nets    

# Apply the UInt64 patch for better JAX compatibility
uint64_patch()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_HYPERPARAMS = {
    "num_timesteps": 100_000,
    "num_envs": 32,
    "episode_length": 256,
    "learning_rate": 1e-3,
    "entropy_cost": 1e-3,
    "discounting": 0.97,
    "unroll_length": 32,
    "batch_size": 1024,
    "num_minibatches": 4,
    "num_updates_per_batch": 2,
    "normalize_observations": True,
    "seed": 0,
    "num_evals": 2,
    "reward_scaling": 1.0,
}

# ---------------------------------------------------------------------------
# Network Factory
# ---------------------------------------------------------------------------
def make_tiny_networks(
    obs_size: int,
    act_size: int,
    preprocess_observations_fn: Callable | None = None
) -> ppo_nets.PPONetworks:
    """
    Create a small PPO network architecture for the passive walker.
    
    Args:
        obs_size: Size of the observation space
        act_size: Size of the action space
        preprocess_observations_fn: Optional observation preprocessing function
        
    Returns:
        PPONetworks: Configured PPO networks
    """
    return ppo_nets.make_ppo_networks(
        observation_size=obs_size,
        action_size=act_size,
        preprocess_observations_fn=preprocess_observations_fn,
        policy_hidden_layer_sizes=(128, 128, 128),
        value_hidden_layer_sizes=(128, 128, 128),
        activation=nn.tanh,
    )

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------
def run_tiny_ppo(
    save_dir: Path,
    walker_sys: brax.System,
    hyperparams: Dict[str, Any] | None = None
) -> Callable:
    """
    Run PPO training for the passive walker.
    
    Args:
        save_dir: Directory to save training artifacts
        walker_sys: Brax system configuration
        hyperparams: Optional hyperparameter overrides
        
    Returns:
        Callable: Trained policy function
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    hp = DEFAULT_HYPERPARAMS.copy()
    if hyperparams:
        hp.update(hyperparams)

    single_env = BraxPassiveWalker(walker_sys)
    print('obs dim:', single_env.observation_size, '| act dim:', single_env.action_size)

    log: List[Tuple[int, Dict]] = []

    def _progress(step: int, metrics: Dict) -> None:
        log.append((step, metrics))
        reward = metrics.get("evaluation/episode_reward",
                           metrics.get("eval/episode_reward",
                                     metrics.get("eval/episode_return", 0.0)))
        print(f"step {step:6d} | eval reward {reward:.2f}")

    print("⇢ launching PPO sanity run …")
    t0 = time.time()
    make_policy, params, _ = ppo_train.train(
        environment=single_env,
        wrap_env=True,
        network_factory=make_tiny_networks,
        progress_fn=_progress,
        **hp,
    )
    elapsed = time.time() - t0
    print(f"✓ training finished in {elapsed:.1f}s")

    # Save training artifacts
    _save_training_artifacts(save_dir, log)
    
    policy_fn = make_policy(params, deterministic=True)
    print("Policy function created...")
    return policy_fn

def _save_training_artifacts(save_dir: Path, log: List[Tuple[int, Dict]]) -> None:
    """Save training metrics and plots."""
    # Save metrics
    with open(save_dir / "tiny_ppo_metrics.pkl", "wb") as f:
        pickle.dump(log, f)

    # Plot and save reward curve
    rewards = [
        m.get("evaluation/episode_reward",
              m.get("eval/episode_reward",
                    m.get("eval/episode_return", 0.0)))
        for _, m in log
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, marker="o")
    plt.grid(True)
    plt.title("Tiny PPO – eval reward")
    plt.xlabel("eval #")
    plt.ylabel("return")
    plt.savefig(save_dir / "tiny_ppo_reward_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plotting reward curve...")

def _run_mujoco_visualization(policy_fn: Callable, sim_time: float) -> None:
    """Run visualization in MuJoCo GUI."""
    mj_env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=sim_time,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True,
        randomize_physics=False
    )

    obs = mj_env.reset()
    key = jax.random.PRNGKey(0)
    done = False
    cum_reward = 0.0
    t0 = time.time()
    mj_env.render()

    print("Starting MuJoCo simulation...")
    while not done and (not mj_env.window or not glfw.window_should_close(mj_env.window)):
        key, sub = jax.random.split(key)
        act_jax, _ = policy_fn(jnp.array(obs), sub)
        act = np.asarray(act_jax, dtype=np.float32)

        obs, reward, done, _ = mj_env.step(act)
        cum_reward += reward
        mj_env.render(mode="human")

    print(f"\nEpisode finished in {time.time()-t0:.1f}s  |  total reward = {cum_reward:.2f}")
    mj_env.close()

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main(mujoco_sim_time: float | None = None) -> None:
    """
    Main execution function.
    
    Args:
        mujoco_sim_time: Optional simulation time for MuJoCo visualization
    """
    # Load and convert MuJoCo model
    if not XML_PATH.exists():
        raise FileNotFoundError(f'XML file not found: {XML_PATH.resolve()}')
    
    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    walker_sys = mjcf.load_model(mj_model)
    print('MuJoCo XML successfully loaded and converted to Brax System')

    # Train policy
    out_dir = RESULTS_BRAX / "tiny_run"
    policy_fn = run_tiny_ppo(out_dir, walker_sys)
    
    # Optional MuJoCo visualization
    if mujoco_sim_time is not None:
        _run_mujoco_visualization(policy_fn, mujoco_sim_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PPO training for passive walker")
    parser.add_argument("--mujoco", type=float, help="MuJoCo GUI episode length in seconds")
    args = parser.parse_args()
    
    main(args.mujoco)
