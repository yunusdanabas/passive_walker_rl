#!/usr/bin/env python
"""
Run the end-to-end PPO training pipeline (no BC) for Passive Walker:
1. Train a PPO policy from scratch.
2. Demo the trained policy in a MuJoCo GUI rollout.

Usage:
    python -m passive_walker.ppo.no_init.run_pipeline [--gpu] [--sim-duration S]
"""

import argparse
import subprocess
import pickle
import numpy as np
import jax.numpy as jnp
import glfw

from . import DATA_DIR, XML_PATH, set_device
from .utils import initialize_policy, load_pickle

def ensure(cmd):
    print("▶", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser(description="Run PPO-from-scratch pipeline (train + demo)")
    p.add_argument("--gpu", action="store_true", help="Use GPU for JAX")
    p.add_argument("--sim-duration", type=float, default=30.0, help="Seconds for the final GUI rollout")
    args = p.parse_args()

    # 0) Configure device
    set_device(args.gpu)

    # 1) Train PPO policy from scratch
    train_cmd = [
        "python", "-m", "passive_walker.ppo.scratch.train",
    ]
    if args.gpu:
        train_cmd.append("--gpu")
    ensure(train_cmd)

    # 2) Load trained policy and critic
    trained_path = DATA_DIR / "trained_policy_with_critic.pkl"
    with open(trained_path, "rb") as f:
        policy, critic = pickle.load(f)
    print(f"Loaded trained policy from {trained_path}")

    # 3) Re-initialize env and get policy fn
    from passive_walker.envs.mujoco_env import PassiveWalkerEnv

    # Infer obs_dim/act_dim for fresh env
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True,
    )

    obs = env.reset()
    total_reward = 0.0
    print(f"[demo] Starting GUI rollout for {args.sim_duration:.1f}s…")
    while not glfw.window_should_close(env.window):
        obs_j = jnp.array(obs, dtype=jnp.float32)
        act = np.array(policy(obs_j))
        obs, rew, done, _ = env.step(act)
        total_reward += rew
        env.render()
        if done:
            break

    env.close()
    print(f"[demo] Episode finished – total reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
