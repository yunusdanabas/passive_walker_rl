#!/usr/bin/env python
"""
Collect on-policy trajectory data using a policy.

This script initializes an environment and a policy from scratch,
then collects trajectory data by running the policy in the environment for a 
specified number of steps. The trajectories are saved to a pickle file for 
later use in training or analysis.
"""

import argparse
import jax.numpy as jnp
import numpy as np

from passive_walker.ppo.scratch import (
    DATA_PPO_SCRATCH, XML_PATH, set_device,
    DEFAULT_STEPS, DEFAULT_SIGMA, DEFAULT_HZ
)
from passive_walker.ppo.scratch.utils import (
    initialize_policy, collect_trajectories, save_pickle
)
from passive_walker.envs.mujoco_env import PassiveWalkerEnv

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Collect on‐policy rollouts")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                  help=f"Env steps per rollout (default: {DEFAULT_STEPS})")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                  help=f"Policy std in scaled space (default: {DEFAULT_SIGMA})")
    p.add_argument("--output", type=str, default="trajectories.pkl",
                  help="Output filename")
    p.add_argument("--hz", type=int, default=DEFAULT_HZ,
                  help=f"Simulation frequency (Hz) (default: {DEFAULT_HZ})")
    p.add_argument("--gpu", action="store_true",
                  help="Use GPU acceleration")
    args = p.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Initialize dummy env to get dimensions
    dummy_env = PassiveWalkerEnv(xml_path=str(XML_PATH), simend=30.0, use_gui=False)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    
    # Initialize policy and environment
    env, get_scaled, get_env, _ = initialize_policy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        xml_path=str(XML_PATH),
        simend=args.steps/args.hz,
        sigma=args.sigma,
        use_gui=False
    )

    # Collect trajectory data using the policy
    traj = collect_trajectories(
        env,
        env_action_fn=get_env,
        scaled_action_fn=get_scaled,
        num_steps=args.steps,
        render=False
    )

    # Save collected trajectories to disk
    out_file = DATA_PPO_SCRATCH / args.output
    save_pickle(traj, out_file)
    print(f"[collect] saved → {out_file}")

if __name__ == "__main__":
    main()
