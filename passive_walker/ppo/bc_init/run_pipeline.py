#!/usr/bin/env python
"""
Main script to run the complete BC-seeded PPO pipeline.

This pipeline:
1. Trains a PPO policy initialized from a Behavioral Cloning (BC) model
2. Loads the trained policy and critic
3. Visualizes the training progress
4. Demonstrates the policy in the passive walker environment

Usage:
    python -m passive_walker.ppo.bc_init.run_pipeline \
        --bc-model hip_knee_mse_controller_20000steps.pkl \
        [--sim-duration D] [--hz HZ] [--gpu] [--no-plot]
"""

import argparse, subprocess, pickle
import numpy as np
import jax.numpy as jnp
from mujoco.glfw import glfw

from passive_walker.ppo.bc_init.utils import initialize_policy, analyze_training_log
from passive_walker.ppo.bc_init.train import Critic
from passive_walker.ppo.bc_init import set_device, PPO_BC_DATA, XML_PATH, BC_DATA

def ensure(cmd):
    """
    Run a command and check that it completes successfully.
    
    Args:
        cmd: List of command arguments
    """
    print("", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--bc-model",    required=True)
    p.add_argument("--gpu",         action="store_true")
    p.add_argument("--sim-duration",type=float, default=30.0)
    p.add_argument("--hz",          type=int,   default=1000)
    p.add_argument("--no-plot",     action="store_true", help="Skip plotting training results")
    args = p.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Construct BC model path
    BC_MODEL = BC_DATA / "hip_knee_mse" / args.bc_model

    # Train PPO agent with critic network
    ensure([
      "python", "-m", "passive_walker.ppo.bc_init.train",
      "--bc-model", args.bc_model,
      "--hz", str(args.hz),
      *(["--gpu"] if args.gpu else [])
    ])

    # Load trained policy and critic
    policy_path = PPO_BC_DATA / f"trained_policy_with_critic_{args.hz}hz.pkl"
    with open(policy_path, "rb") as f:
        policy, critic = pickle.load(f)
        
    # Analyze and plot training results
    if not args.no_plot:
        analyze_training_log(log_path=PPO_BC_DATA / f"ppo_training_log_{args.hz}hz.pkl",
                           save_path=PPO_BC_DATA / f"ppo_training_curve_{args.hz}hz.png")

    # Demo the trained policy in the environment
    env, get_scaled, get_env, _ = initialize_policy(
        model_path=str(BC_MODEL),
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        sigma=0.1, 
        use_gui=True
    )
    
    # Run the environment loop
    o, total = env.reset(), 0.0
    done = False
    try:
        while not done:
            aj = jnp.array(o, dtype=jnp.float32)
            act = get_env(aj)
            o, r, done, _ = env.step(act)
            total += r
            env.render()  # ensures window exists
            if env.window is not None and glfw.window_should_close(env.window):
                break
    finally:
        # Ensure proper cleanup
        if env.window is not None:
            env.close()
    print(f"[demo] total reward = {total:.2f}")

if __name__=="__main__":
    main()
