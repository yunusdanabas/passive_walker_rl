#!/usr/bin/env python
"""
Main script to run the complete BC-seeded PPO pipeline.

This pipeline:
1. Trains a PPO policy initialized from a Behavioral Cloning (BC) model
2. Loads the trained policy and critic
3. Visualizes the training progress
4. Demonstrates the policy in the passive walker environment
"""

import argparse, subprocess, pickle
import numpy as np
import jax.numpy as jnp
from mujoco.glfw import glfw

from passive_walker.ppo.bc_init.utils import initialize_policy, analyze_training_log
from passive_walker.ppo.bc_init.train import Critic
from . import set_device, DATA_DIR, XML_PATH

def ensure(cmd):
    """
    Run a command and check that it completes successfully.
    
    Args:
        cmd: List of command arguments
    """
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # Parse command line arguments
    p = argparse.ArgumentParser()
    p.add_argument("--bc-model",    required=True)
    p.add_argument("--gpu",         action="store_true")
    p.add_argument("--sim-duration",type=float, default=30.0)
    p.add_argument("--no-plot",     action="store_true", help="Skip plotting training results")
    args = p.parse_args()

    # Set device (CPU/GPU)
    set_device(args.gpu)

    # Train PPO agent with critic network
    ensure([
      "python", "-m", "passive_walker.ppo.bc_init.train",
      "--bc-model", args.bc_model,
      *(["--gpu"] if args.gpu else [])
    ])

    # Load trained policy and critic
    policy_path = DATA_DIR / "trained_policy_with_critic.pkl"
    with open(policy_path, "rb") as f:
        policy, critic = pickle.load(f)
        
    # Analyze and plot training results
    if not args.no_plot:
        analyze_training_log()  # Will use default paths

    # Demo the trained policy in the environment
    env, get_scaled, get_env, _ = initialize_policy(
        model_path=args.bc_model,
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        sigma=0.1, 
        use_gui=True
    )
    
    # Run the environment loop
    o = env.reset()
    total=0.0
    done=False
    while not done and not glfw.window_should_close(env.window):
        aj = jnp.array(o, dtype=jnp.float32)
        act = get_env(aj)
        no, r, done, _ = env.step(act)
        total += r
        env.render()
        o = no
    env.close()
    print(f"[demo] total reward = {total:.2f}")

if __name__=="__main__":
    main()
