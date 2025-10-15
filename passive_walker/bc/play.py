"""
BC Model Evaluation and Playback

Unified evaluation CLI supporting both PyTorch and JAX backends for trained BC models.
Supports different control sections and provides detailed performance metrics.
"""

from __future__ import annotations
import argparse
import sys
import os
import json
import time
import numpy as np
from passive_walker.core.env import PassiveWalkerEnv
from .utils import set_seed, Normalizer


def _norm_to_action(qdes: float, lo: float, hi: float) -> float:
    """Map physical qdes -> normalized [-1,1] action space."""
    return float(2.0 * (qdes - lo) / (hi - lo) - 1.0)


def _assemble_action_torch(section: str, model_out: np.ndarray, fsm, data, model_mj):
    """
    Build full 3D action vector for PyTorch models.
    
    Combines BC model output with FSM control for unmanaged joints.
    """
    # Import here to avoid circular imports
    from passive_walker.core.controller import JOINT_MIN, JOINT_MAX
    import mujoco
    
    hip_lo, lk_lo, rk_lo = JOINT_MIN.tolist()
    hip_hi, lk_hi, rk_hi = JOINT_MAX.tolist()

    # Get body IDs for FSM contact detection
    b_lfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    b_rfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    
    # Update FSM to get physical desireds for joints we don't control
    fsm.update(data, model_mj, b_lfoot, b_rfoot)
    hip_fsm = fsm.desired_hip()
    lk_fsm, rk_fsm = fsm.desired_knees()

    if section == "hip":
        # BC controls hip, FSM controls knees
        hip = float(model_out[0])
        lk = _norm_to_action(lk_fsm, lk_lo, lk_hi)
        rk = _norm_to_action(rk_fsm, rk_lo, rk_hi)
        return np.array([hip, lk, rk], dtype=np.float32)

    if section == "knees":
        # FSM controls hip, BC controls knees
        hip = _norm_to_action(hip_fsm, hip_lo, hip_hi)
        lk, rk = map(float, model_out[:2])
        return np.array([hip, lk, rk], dtype=np.float32)

    # both / both-adv: BC controls all joints
    # Normalize model output from [-1,1] to physical joint ranges
    hip = _norm_to_action(float(model_out[0]), hip_lo, hip_hi)
    lk = _norm_to_action(float(model_out[1]), lk_lo, lk_hi)
    rk = _norm_to_action(float(model_out[2]), rk_lo, rk_hi)
    return np.array([hip, lk, rk], dtype=np.float32)


def _assemble_action_jax(section: str, model_out: np.ndarray, fsm, data, model_mj):
    """
    Build full 3D action vector for JAX models.
    
    Combines BC model output with FSM control for unmanaged joints.
    """
    # Import here to avoid circular imports
    from passive_walker.core.controller import JOINT_MIN, JOINT_MAX
    import mujoco
    
    hip_lo, lk_lo, rk_lo = JOINT_MIN.tolist()
    hip_hi, lk_hi, rk_hi = JOINT_MAX.tolist()

    # Get body IDs for FSM contact detection
    b_lfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    b_rfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    
    # Update FSM to get physical desireds for joints we don't control
    fsm.update(data, model_mj, b_lfoot, b_rfoot)
    hip_fsm = fsm.desired_hip()
    lk_fsm, rk_fsm = fsm.desired_knees()

    if section == "hip":
        # BC controls hip, FSM controls knees
        hip = float(model_out[0])
        lk = _norm_to_action(lk_fsm, lk_lo, lk_hi)
        rk = _norm_to_action(rk_fsm, rk_lo, rk_hi)
        return np.array([hip, lk, rk], dtype=np.float32)

    if section == "knees":
        # FSM controls hip, BC controls knees
        hip = _norm_to_action(hip_fsm, hip_lo, hip_hi)
        lk, rk = map(float, model_out[:2])
        return np.array([hip, lk, rk], dtype=np.float32)

    # both / both-adv: BC controls all joints
    # Normalize model output from [-1,1] to physical joint ranges
    hip = _norm_to_action(float(model_out[0]), hip_lo, hip_hi)
    lk = _norm_to_action(float(model_out[1]), lk_lo, lk_hi)
    rk = _norm_to_action(float(model_out[2]), rk_lo, rk_hi)
    return np.array([hip, lk, rk], dtype=np.float32)


def play_torch(ckpt_path: str, meta_path: str, episodes: int, seconds: float, seed: int, headless: bool, frame_stack: int = 1):
    """
    Play PyTorch BC model with GUI and performance metrics.
    
    Args:
        ckpt_path: Path to model checkpoint (.pt file)
        meta_path: Path to metadata (.json file)
        episodes: Number of episodes to run
        seconds: Max seconds per episode
        seed: Random seed
        headless: If True, disable GUI
    """
    import torch
    from .models.models_torch import TorchMLP, TorchMLPLarge
    from .utils import load_checkpoint
    from passive_walker.core.controller import FSMStateMachine

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Create model and load weights (try large model first, fallback to small)
    try:
        model = TorchMLPLarge(in_dim=meta["input_dim"], out_dim=meta["output_dim"], hidden=512, dropout=0.1)
        model = load_checkpoint(ckpt_path, model)
    except Exception:
        # Fallback to small model if large model fails to load
        model = TorchMLP(in_dim=meta["input_dim"], out_dim=meta["output_dim"])
        model = load_checkpoint(ckpt_path, model)
    
    model.eval()

    # Setup normalizer
    normalizer = Normalizer(
        mean=np.array(meta["normalizer_mean"], dtype=np.float32),
        std=np.array(meta["normalizer_std"], dtype=np.float32)
    )

    # Create environment
    env = PassiveWalkerEnv(mode="research", use_gui=not headless)
    fsm = FSMStateMachine()
    
    # Bind FSM indices for performance
    fsm.bind_indices(
        qpos_hip=env.qpos_hip, qpos_lk=env.qpos_lk, qpos_rk=env.qpos_rk,
        qvel_hip=env.qvel_hip, qvel_lk=env.qvel_lk, qvel_rk=env.qvel_rk,
        b_lfoot=env.b_lfoot, b_rfoot=env.b_rfoot,
        b_lleg=env.b_lleg, b_rleg=env.b_rleg
    )

    # Performance tracking
    results = {
        "episodes": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "total_time": 0.0,
        "falls": 0
    }

    print(f"[INFO] Playing {episodes} episodes (max {seconds}s each, seed={seed})")
    print(f"[INFO] Model: {meta['section']} section, {meta['input_dim']}D input, {meta['output_dim']}D output")

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0
        episode_steps = 0
        start_time = time.time()

        print(f"\n--- Episode {ep + 1}/{episodes} ---")

        try:
            while env.data.time < seconds:
                # Get model prediction
                if frame_stack > 1:
                    # For frame stacking, we need to maintain a buffer of observations
                    if not hasattr(play_torch, 'obs_buffer'):
                        play_torch.obs_buffer = []
                    
                    play_torch.obs_buffer.append(obs)
                    if len(play_torch.obs_buffer) > frame_stack:
                        play_torch.obs_buffer.pop(0)
                    
                    if len(play_torch.obs_buffer) < frame_stack:
                        # Not enough frames yet, use FSM only
                        action = _assemble_action_torch(meta["section"], np.array([0.0]), fsm, env.data, env.model)
                    else:
                        # Stack frames
                        x = np.concatenate(play_torch.obs_buffer).astype(np.float32)
                        x = normalizer.apply(x[None, :]).astype(np.float32)
                        with torch.no_grad():
                            model_out = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
                        action = _assemble_action_torch(meta["section"], model_out, fsm, env.data, env.model)
                else:
                    # No frame stacking
                    x = normalizer.apply(obs[None, :]).astype(np.float32)
                    with torch.no_grad():
                        model_out = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
                    action = _assemble_action_torch(meta["section"], model_out, fsm, env.data, env.model)

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Print progress
                if episode_steps % 50 == 0:
                    print(f"  t={env.data.time:6.2f}s  dx={info.get('dx', 0):+.4f}  "
                          f"pitch={info.get('pitch_abs', 0):.3f}  r={reward:+.4f}")

                if done:
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            break

        # Episode summary
        episode_time = time.time() - start_time
        fell = info.get("fell", False)
        
        print(f"  Episode {ep + 1} complete: {episode_steps} steps, {episode_time:.1f}s, "
              f"reward={episode_reward:+.2f}, fell={fell}")

        # Update results
        results["episodes"].append({
            "steps": episode_steps,
            "reward": episode_reward,
            "time": episode_time,
            "fell": fell
        })
        results["total_reward"] += episode_reward
        results["total_steps"] += episode_steps
        results["total_time"] += episode_time
        if fell:
            results["falls"] += 1

    # Final summary
    env.close()
    avg_reward = results["total_reward"] / max(1, len(results["episodes"]))
    avg_steps = results["total_steps"] / max(1, len(results["episodes"]))
    success_rate = (len(results["episodes"]) - results["falls"]) / max(1, len(results["episodes"]))
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Episodes: {len(results['episodes'])}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:+.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Total time: {results['total_time']:.1f}s")
    
    return results


def play_jax(ckpt_path: str, meta_path: str, episodes: int, seconds: float, seed: int, headless: bool, frame_stack: int = 1):
    """
    Play JAX BC model with GUI and performance metrics.
    
    Args:
        ckpt_path: Path to model checkpoint (.eqx file)
        meta_path: Path to metadata (.json file)
        episodes: Number of episodes to run
        seconds: Max seconds per episode
        seed: Random seed
        headless: If True, disable GUI
    """
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from .models.models_jax import make_model, load_eqx
    from passive_walker.core.controller import FSMStateMachine

    # Load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Create model and load weights
    key = jax.random.PRNGKey(seed)
    model = make_model(
        in_dim=meta["in_dim"], 
        out_dim=meta["out_dim"], 
        width=meta.get("hidden", 128), 
        depth=meta.get("depth", 2), 
        key=key
    )
    model = load_eqx(ckpt_path, model)

    # Setup normalizer
    normalizer = Normalizer(
        mean=np.array(meta["normalizer"]["mean"], dtype=np.float32),
        std=np.array(meta["normalizer"]["std"], dtype=np.float32)
    )

    # Create environment
    env = PassiveWalkerEnv(mode="research", use_gui=not headless)
    fsm = FSMStateMachine()
    
    # Bind FSM indices for performance
    fsm.bind_indices(
        qpos_hip=env.qpos_hip, qpos_lk=env.qpos_lk, qpos_rk=env.qpos_rk,
        qvel_hip=env.qvel_hip, qvel_lk=env.qvel_lk, qvel_rk=env.qvel_rk,
        b_lfoot=env.b_lfoot, b_rfoot=env.b_rfoot,
        b_lleg=env.b_lleg, b_rleg=env.b_rleg
    )

    # Performance tracking
    results = {
        "episodes": [],
        "total_reward": 0.0,
        "total_steps": 0,
        "total_time": 0.0,
        "falls": 0
    }

    print(f"[INFO] Playing {episodes} episodes (max {seconds}s each, seed={seed})")
    print(f"[INFO] Model: {meta['section']} section, {meta['in_dim']}D input, {meta['out_dim']}D output")

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0
        episode_steps = 0
        start_time = time.time()

        print(f"\n--- Episode {ep + 1}/{episodes} ---")

        try:
            while env.data.time < seconds:
                # Get model prediction
                x = normalizer.apply(obs[None, :]).astype(np.float32)
                x_jax = jnp.asarray(x)
                model_out = np.asarray(model(x_jax))

                # Assemble full action
                action = _assemble_action_jax(meta["section"], model_out, fsm, env.data, env.model)

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Print progress
                if episode_steps % 50 == 0:
                    print(f"  t={env.data.time:6.2f}s  dx={info.get('dx', 0):+.4f}  "
                          f"pitch={info.get('pitch_abs', 0):.3f}  r={reward:+.4f}")

                if done:
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
            break

        # Episode summary
        episode_time = time.time() - start_time
        fell = info.get("fell", False)
        
        print(f"  Episode {ep + 1} complete: {episode_steps} steps, {episode_time:.1f}s, "
              f"reward={episode_reward:+.2f}, fell={fell}")

        # Update results
        results["episodes"].append({
            "steps": episode_steps,
            "reward": episode_reward,
            "time": episode_time,
            "fell": fell
        })
        results["total_reward"] += episode_reward
        results["total_steps"] += episode_steps
        results["total_time"] += episode_time
        if fell:
            results["falls"] += 1

    # Final summary
    env.close()
    avg_reward = results["total_reward"] / max(1, len(results["episodes"]))
    avg_steps = results["total_steps"] / max(1, len(results["episodes"]))
    success_rate = (len(results["episodes"]) - results["falls"]) / max(1, len(results["episodes"]))
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Episodes: {len(results['episodes'])}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:+.2f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Total time: {results['total_time']:.1f}s")
    
    return results


def main():
    """Main CLI entry point for BC model evaluation."""
    parser = argparse.ArgumentParser("BC play")
    parser.add_argument("--ckpt", required=True, help="model checkpoint (.pt or .eqx)")
    parser.add_argument("--meta", required=True, help="metadata (.json)")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--seconds", type=float, default=25.0, help="max seconds per episode")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--gui", action="store_true", help="enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="disable GUI")
    parser.set_defaults(gui=True)
    parser.add_argument("--frame-stack", type=int, default=1, help="Number of frames to stack for temporal context")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # Determine backend from file extension
    if args.ckpt.endswith(".pt"):
        backend = "torch"
    elif args.ckpt.endswith(".eqx"):
        backend = "jax"
    else:
        sys.exit(f"Unknown checkpoint format: {args.ckpt}")

    print(f"[OK] Backend={backend}  Episodes={args.episodes}  Seconds={args.seconds}  GUI={args.gui}")

    if backend == "torch":
        try: 
            import torch  # noqa
            results = play_torch(args.ckpt, args.meta, args.episodes, args.seconds, args.seed or 0, not args.gui, args.frame_stack)
        except Exception as e: 
            sys.exit(f"Torch backend error: {e}")
    else:
        try: 
            import jax  # noqa
            results = play_jax(args.ckpt, args.meta, args.episodes, args.seconds, args.seed or 0, not args.gui, args.frame_stack)
        except Exception as e: 
            sys.exit(f"JAX backend error: {e}")


if __name__ == "__main__":
    main()