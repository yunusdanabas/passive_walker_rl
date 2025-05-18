#!/usr/bin/env python
"""
Collect on-policy trajectory data using a behavior-cloned policy.

This script initializes an environment and loads a behavior-cloned (BC) policy,
then collects trajectory data by running the policy in the environment for a 
specified number of steps. The trajectories are saved to a pickle file for 
later use in training.
"""

import argparse
import jax.numpy as jnp
import numpy as np

from . import DATA_DIR, XML_PATH, set_device
from .utils import initialize_policy, collect_trajectories, save_pickle

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Collect on‐policy rollouts")
    p.add_argument("--bc-model", type=str,    required=True,
                   help="BC‐trained controller (.pkl)")
    p.add_argument("--steps",    type=int,    default=4096,
                   help="Env steps per rollout")
    p.add_argument("--sigma",    type=float,  default=0.1,
                   help="Policy std in scaled space")
    p.add_argument("--gpu",      action="store_true")
    args = p.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Initialize environment and policy from BC model
    env, get_scaled, get_env, _ = initialize_policy(
        model_path=str(args.bc_model),
        xml_path=str(XML_PATH),
        simend=args.steps/60.0,  # Convert steps to simulation time
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
    out_file = DATA_DIR / "trajectories.pkl"
    save_pickle(traj, out_file)
    print(f"[collect] saved → {out_file}")

if __name__ == "__main__":
    main()
