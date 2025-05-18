#!/usr/bin/env python
"""
Evaluate a trained BC-seeded PPO policy.

This script:
1. Loads a trained PPO policy (with or without critic)
2. Runs the policy in the passive walker environment
3. Visualizes joint positions and rewards over time
"""

import os
import argparse
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mujoco.glfw import glfw

from . import XML_PATH, DATA_DIR, set_device
from .utils import load_pickle
from passive_walker.envs.mujoco_env import PassiveWalkerEnv

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Evaluate BC‐seeded PPO policy")
    p.add_argument("--policy",       type=str,              required=True)
    p.add_argument("--sim-duration", type=float, default=30.0)
    p.add_argument("--gpu",          action="store_true")
    args = p.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Load policy (ignore critic if present)
    policy, critic = load_pickle(os.path.join(DATA_DIR, args.policy))

    # Initialize environment
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True,
    )

    # Run evaluation loop
    obs = env.reset()
    traj_obs, rewards = [], []

    while not glfw.window_should_close(env.window):
        traj_obs.append(obs)
        obs_j = jnp.array(obs, dtype=jnp.float32)
        act  = np.array(policy(obs_j))
        obs, r, done, _ = env.step(act)
        rewards.append(r)
        env.render()
        if done:
            break

    env.close()

    # Convert to numpy arrays for plotting
    traj_obs = np.array(traj_obs)
    rewards  = np.array(rewards)
    t = np.arange(len(rewards))

    # Plot joint positions
    plt.figure(figsize=(8,4))
    for i in range(min(3, traj_obs.shape[1]//2)):
        plt.plot(t, traj_obs[:,i], label=f"joint {i}")
    plt.title("Joint Positions"); plt.legend(); plt.show()

    # Plot rewards
    plt.figure(figsize=(8,3))
    plt.plot(t, rewards); plt.title("Reward per step"); plt.show()

if __name__ == "__main__":
    main()
