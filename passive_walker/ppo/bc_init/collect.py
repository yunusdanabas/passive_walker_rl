#!/usr/bin/env python
"""
Collect on-policy trajectory data using a behavior-cloned policy.
Saves trajectories to a pickle file for later use in training.
"""

import argparse
import jax.numpy as jnp
import numpy as np

from passive_walker.ppo.bc_init import DATA_PPO_BC, XML_PATH, BC_DATA, set_device
from passive_walker.ppo.bc_init.utils import initialize_policy, collect_trajectories, save_pickle

# Default paths and parameters
DEFAULT_BC_MODEL = str(BC_DATA / "hip_knee_mse" / "hip_knee_mse_controller_20000steps.pkl")
DEFAULT_STEPS = 4096
DEFAULT_SIGMA = 0.1

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser(description="Collect on-policy rollouts")
    p.add_argument("--bc-model", type=str, default=DEFAULT_BC_MODEL,
                   help="BC-trained controller path (default: %(default)s)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help="Environment steps per rollout (default: %(default)s)")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                   help="Policy standard deviation (default: %(default)s)")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU if available")
    p.add_argument("--hz", type=int, default=200,
                   help="Simulation frequency (Hz)")
    args = p.parse_args()

    # Setup device and initialize environment
    set_device(args.gpu)
    env, get_scaled, get_env, _ = initialize_policy(
        model_path=args.bc_model,
        xml_path=str(XML_PATH),
        simend=args.steps/args.hz,
        sigma=args.sigma,
        use_gui=False
    )

    # Collect and save trajectories
    traj = collect_trajectories(
        env,
        env_action_fn=get_env,
        scaled_action_fn=get_scaled,
        num_steps=args.steps,
        render=False
    )

    out_file = DATA_PPO_BC / "trajectories.pkl"
    save_pickle(traj, out_file)
    print(f"[collect] Saved trajectories to {out_file}")

if __name__ == "__main__":
    main()
