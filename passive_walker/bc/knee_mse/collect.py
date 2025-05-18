# passive_walker/bc/knee_mse/collect.py
"""
Collect knee-only FSM demos (both knees) for MSE-BC training.

Usage:
    python -m passive_walker.bc.knee_mse.collect [--steps N] [--gpu]
"""

import argparse
import numpy as np
import jax.numpy as jnp

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.utils.io import save_pickle
from . import DATA_DIR, XML_PATH, set_device

def collect_demo_data(env, num_steps: int = 1000):
    """
    Runs the env in FSM demo mode to collect (obs, [left_knee, right_knee]) pairs.
    Returns:
      obs_array   : jnp.ndarray, shape (num_steps, obs_dim)
      label_array : jnp.ndarray, shape (num_steps, 2)
    """
    obs_buf = []
    lbl_buf = []
    obs = env.reset()
    for _ in range(num_steps):
        obs_buf.append(obs)
        left_k = float(env.data.ctrl[env.left_knee_pos_actuator_id])
        right_k= float(env.data.ctrl[env.right_knee_pos_actuator_id])
        lbl_buf.append([left_k, right_k])

        # dummy step (FSM logic still runs inside env)
        dummy_act = np.zeros(3, dtype=np.float32)
        obs, _, done, _ = env.step(dummy_act)
        if done:
            obs = env.reset()

    return jnp.array(obs_buf), jnp.array(lbl_buf)

def main():
    p = argparse.ArgumentParser(
        description="Collect FSM knee demos for BC training"
    )
    p.add_argument(
        "--steps", type=int, default=20_000,
        help="Number of simulation steps to collect"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Run on GPU if available (JAX_PLATFORM_NAME=gpu)"
    )
    args = p.parse_args()

    # Configure device before any JAX imports
    set_device(args.gpu)

    # Build environment: hip still FSM, knees FSM output recorded externally
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.steps / 60.0,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=False
    )

    print(f"[collect] running {args.steps} steps…")
    demo_obs, demo_labels = collect_demo_data(env, num_steps=args.steps)
    env.close()

    print(f"[collect] collected obs={demo_obs.shape}, labels={demo_labels.shape}")

    out_file = DATA_DIR / "knee_mse_demos.pkl"
    save_pickle(
        {"obs": np.array(demo_obs,   dtype=np.float32),
         "labels": np.array(demo_labels, dtype=np.float32)},
        out_file
    )
    print(f"[collect] saved → {out_file}")

if __name__ == "__main__":
    main()
