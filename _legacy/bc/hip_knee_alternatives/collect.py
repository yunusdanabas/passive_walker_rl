"""
Collect hip-knee FSM demos for BC (records hip, left-knee, right-knee).

This script runs the environment in FSM demo mode (hip and knees controlled by FSM)
and records observations along with the FSM actions at each step.

The environment uses a Finite State Machine (FSM) to control both hip and knee joints.
The collected data consists of state observations and corresponding actions that can be
used for behavioral cloning training.

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.collect [--steps N] [--hz H] [--gpu]
"""

import argparse
import numpy as np
import jax.numpy as jnp

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.utils.io import save_pickle
from passive_walker.bc.hip_knee_alternatives import DATA_BC_HIP_KNEE_ALTERNATIVES, XML_PATH, set_device


def collect_demo_data(env: PassiveWalkerEnv, num_steps: int):
    """
    Collect (obs, [hip, left_knee, right_knee]) pairs from the FSM controller.

    Args:
        env:           PassiveWalkerEnv configured with use_nn_for_hip=False, use_nn_for_knees=False
        num_steps:     Total simulation steps to record

    Returns:
        obs_array:     jnp.ndarray of shape (num_steps, obs_dim)
        label_array:   jnp.ndarray of shape (num_steps, 3)
    """
    obs_buf, lbl_buf = [], []
    obs = env.reset()
    for _ in range(num_steps):
        obs_buf.append(obs)
        lbl_buf.append([
            float(env.data.ctrl[env.hip_pos_actuator_id]),
            float(env.data.ctrl[env.left_knee_pos_actuator_id]),
            float(env.data.ctrl[env.right_knee_pos_actuator_id]),
        ])
        obs, _, done, _ = env.step(np.zeros(3, dtype=np.float32))
        if done:
            obs = env.reset()
    return jnp.array(obs_buf), jnp.array(lbl_buf)


def main():
    """Main function to collect demonstration data."""
    p = argparse.ArgumentParser(
        description="Collect FSM hip+knee demos for BC training"
    )
    p.add_argument(
        "--steps", type=int, default=50_000,
        help="Number of simulation steps to collect"
    )
    p.add_argument(
        "--hz", type=int, default=200,
        help="Simulation frequency in Hz"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available (sets JAX_PLATFORM_NAME=gpu)"
    )
    args = p.parse_args()

    # Check for existing file with same step count
    out_file = DATA_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_demos_{args.steps}steps.pkl"
    if out_file.exists():
        print(f"[collect] Found existing file with {args.steps} steps → {out_file}")
        return

    # configure JAX backend
    set_device(args.gpu)

    # build the FSM-only environment
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.steps / args.hz,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=False,
    )

    print(f"[collect] running {args.steps} steps…")
    demo_obs, demo_labels = collect_demo_data(env, num_steps=args.steps)
    env.close()

    print(f"[collect] collected obs={demo_obs.shape}, labels={demo_labels.shape}")

    save_pickle(
        {"obs": np.array(demo_obs, dtype=np.float32),
         "labels": np.array(demo_labels, dtype=np.float32)},
        out_file
    )
    print(f"[collect] saved → {out_file}")


if __name__ == "__main__":
    main()
