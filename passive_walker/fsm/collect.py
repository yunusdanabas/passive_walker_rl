"""
FSM Data Collection (Ultra-Lean)

High-performance FSM data collection with pre-allocation, deterministic seeding,
and clean NPZ schema for BC consumption.

Parameters live at the top of this file; no YAML config needed.
GUI is off by default; see passive_walker/fsm/README.md for usage.
"""
import os
import json
import time
import argparse
import numpy as np
from passive_walker.core.env import PassiveWalkerEnv

# ---------- knobs (edit here) ----------
DEFAULT_EPISODES = 10        # episodes to collect
DEFAULT_STEPS = 512          # steps per episode (upper bound; episode may end earlier)
DEFAULT_OUTDIR = "data/fsm_runs"  # output directory
DEFAULT_SEED = 123           # base random seed
DEFAULT_MODE = "fsm"         # don't change: collector assumes FSM
PRINT_EVERY_SEC = 0.5        # console throttle
SAVE_META = True             # write <outdir>/meta.json once
COMPRESS_NPZ = True          # use np.savez_compressed
# --------------------------------------


def _save_npz(path, payload, compress=True):
    """Save NPZ file with optional compression."""
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def collect(episodes, steps, outdir, seed=None):
    """
    Collect FSM walking episodes with pre-allocation and deterministic seeding.
    
    Args:
        episodes: Number of episodes to collect
        steps: Maximum steps per episode
        outdir: Output directory for NPZ files
        seed: Base random seed (None for non-deterministic)
    """
    os.makedirs(outdir, exist_ok=True)

    # One env reused across episodes
    env = PassiveWalkerEnv(mode=DEFAULT_MODE, use_gui=False)

    # Save metadata once
    if SAVE_META:
        meta = {
            "episodes": int(episodes),
            "steps_per_episode": int(steps),
            "seed": None if seed is None else int(seed),
            "env": "PassiveWalkerEnv",
            "mode": "fsm",
            "pd_backend": env.pd.backend_name,
            "ctrl_hz": env.ctrl_hz,
            "schema": {
                "obs": "(T+1,11)",
                "act": "(T,3)",
                "rew": "(T,)",
                "done": "(T,)",
                "info_pitch": "(T,)",
                "info_torso_z": "(T,)",
                "info_dx": "(T,)"
            },
        }
        with open(os.path.join(outdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    for ep in range(episodes):
        # Deterministic per-episode seed (nice for reproducibility & sharding)
        ep_seed = None if seed is None else (int(seed) + ep)
        obs, _ = env.reset(seed=ep_seed)

        # Pre-allocate buffers (fast, no per-step allocation)
        T = int(steps)
        obs_buf = np.empty((T + 1, 11), dtype=np.float32)
        obs_buf[0] = obs
        act_buf = np.zeros((T, 3), dtype=np.float32)      # FSM ignores actions
        rew_buf = np.empty((T,), dtype=np.float32)
        done_buf = np.empty((T,), dtype=np.bool_)
        ipitch = np.empty((T,), dtype=np.float32)
        itorso = np.empty((T,), dtype=np.float32)
        idx = np.empty((T,), dtype=np.float32)

        last_print = time.time()
        t_used = 0

        for t in range(T):
            # Action zeros (FSM mode)
            action = act_buf[t]
            obs, r, done, info = env.step(action)

            # Write row
            obs_buf[t + 1] = obs
            rew_buf[t] = r
            done_buf[t] = done
            ipitch[t] = info.get("pitch_abs", 0.0)
            itorso[t] = info.get("torso_z", 0.0)
            idx[t] = info.get("dx", 0.0)
            t_used = t + 1

            # Periodic print
            now = time.time()
            if (now - last_print) >= PRINT_EVERY_SEC:
                print(f"[ep {ep+1}/{episodes}] t={t_used:4d}/{T} "
                      f"dx={idx[t]:+.4f} pitch|rad|={ipitch[t]:.3f} "
                      f"torso_z={itorso[t]:.3f} done={bool(done)}")
                last_print = now

            if done:
                break

        # Trim to actual length and save
        payload = dict(
            obs=obs_buf[:t_used + 1],
            act=act_buf[:t_used],
            rew=rew_buf[:t_used],
            done=done_buf[:t_used],
            info_pitch=ipitch[:t_used],
            info_torso_z=itorso[:t_used],
            info_dx=idx[:t_used],
        )
        out_path = os.path.join(outdir, f"episode_{ep:06d}.npz")
        _save_npz(out_path, payload, compress=COMPRESS_NPZ)
        print(f"saved: {out_path} (T={t_used})")

    env.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser("FSM data collection (ultra-lean)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="Steps per episode")
    parser.add_argument("--out", type=str, default=DEFAULT_OUTDIR,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    args = parser.parse_args()
    
    collect(args.episodes, args.steps, args.out, args.seed)


if __name__ == "__main__":
    main()