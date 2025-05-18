"""
Collect hip-knee FSM demos for BC (records hip, left-knee, right-knee).

Usage:
    python -m passive_walker.bc.hip_knee_mse.collect [--steps N] [--gpu]
"""

import argparse
import numpy as np
import jax.numpy as jnp

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.utils.io            import save_pickle
from . import DATA_DIR, XML_PATH, set_device


def collect_demo_data(env, num_steps: int):
    """Return (obs, labels) where labels = [hip, left_knee, right_knee]."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--gpu",   action="store_true")
    args = p.parse_args()

    set_device(args.gpu)

    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.steps / 60.0,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=False,
    )

    print(f"[collect] running {args.steps} steps …")
    obs, labels = collect_demo_data(env, args.steps)
    env.close()
    print(f"[collect] obs={obs.shape}, labels={labels.shape}")

    out = DATA_DIR / "hip_knee_mse_demos.pkl"
    save_pickle({"obs": np.array(obs,    dtype=np.float32),
                 "labels": np.array(labels, dtype=np.float32)}, out)
    print(f"[collect] saved → {out}")


if __name__ == "__main__":
    main()
