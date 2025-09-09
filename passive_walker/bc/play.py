"""
Play/evaluate CLI with full Torch and JAX support.
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
    """Map physical qdes -> normalized [-1,1]."""
    return float(2.0 * (qdes - lo) / (hi - lo) - 1.0)


def _assemble_action_torch(section: str, model_out: np.ndarray, fsm, data, model_mj):
    """Build full 3D action vector for Torch models."""
    # Import here to avoid circular imports
    from passive_walker.core.controller import JOINT_MIN, JOINT_MAX
    import mujoco
    
    hip_lo, lk_lo, rk_lo = JOINT_MIN.tolist()
    hip_hi, lk_hi, rk_hi = JOINT_MAX.tolist()

    # Get body IDs
    b_lfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    b_rfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    
    # Update FSM to get physical desireds for joints we don't control
    fsm.update(data, model_mj, b_lfoot, b_rfoot)
    hip_fsm = fsm.desired_hip()
    lk_fsm, rk_fsm = fsm.desired_knees()

    if section == "hip":
        hip = float(model_out[0])
        lk = _norm_to_action(lk_fsm, lk_lo, lk_hi)
        rk = _norm_to_action(rk_fsm, rk_lo, rk_hi)
        return np.array([hip, lk, rk], dtype=np.float32)

    if section == "knees":
        hip = _norm_to_action(hip_fsm, hip_lo, hip_hi)
        lk, rk = map(float, model_out[:2])
        return np.array([hip, lk, rk], dtype=np.float32)

    # both / both-adv
    return np.array(model_out[:3], dtype=np.float32)


def _assemble_action_jax(section: str, model_out: np.ndarray, fsm, data, model_mj):
    """Build full 3D action vector for JAX models."""
    # Import here to avoid circular imports
    from passive_walker.core.controller import JOINT_MIN, JOINT_MAX
    import mujoco
    
    hip_lo, lk_lo, rk_lo = JOINT_MIN.tolist()
    hip_hi, lk_hi, rk_hi = JOINT_MAX.tolist()

    # Get body IDs
    b_lfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    b_rfoot = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
    
    # Update FSM to get physical desireds for joints we don't control
    fsm.update(data, model_mj, b_lfoot, b_rfoot)
    hip_fsm = fsm.desired_hip()
    lk_fsm, rk_fsm = fsm.desired_knees()

    if section == "hip":
        hip = float(model_out[0])
        lk = _norm_to_action(lk_fsm, lk_lo, lk_hi)
        rk = _norm_to_action(rk_fsm, rk_lo, rk_hi)
        return np.array([hip, lk, rk], dtype=np.float32)

    if section == "knees":
        hip = _norm_to_action(hip_fsm, hip_lo, hip_hi)
        lk, rk = map(float, model_out[:2])
        return np.array([hip, lk, rk], dtype=np.float32)

    # both / both-adv
    return np.array(model_out[:3], dtype=np.float32)


def play_torch(ckpt_path: str, meta_path: str, episodes: int, seconds: float, seed: int, headless: bool = True):
    """Play with Torch model."""
    import torch
    from .models_torch import TorchMLP, TorchMLPLarge
    from .utils import load_checkpoint
    from passive_walker.core.controller import FSMStateMachine
    
    # Load model + meta
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    section = meta["section"]
    normalizer = Normalizer(
        mean=np.array(meta["normalizer_mean"], dtype=np.float32),
        std=np.array(meta["normalizer_std"], dtype=np.float32)
    )
    
    # Create model and load weights (try large model first, fallback to small)
    try:
        model = TorchMLPLarge(in_dim=meta["input_dim"], out_dim=meta["output_dim"], hidden=512, dropout=0.1)
        model = load_checkpoint(ckpt_path, model)
    except Exception:
        # Fallback to small model if large model fails to load
        model = TorchMLP(in_dim=meta["input_dim"], out_dim=meta["output_dim"])
        model = load_checkpoint(ckpt_path, model)
    model.eval()
    
    env = PassiveWalkerEnv(mode="research", use_gui=not headless)
    fsm = FSMStateMachine()
    
    rng = np.random.RandomState(seed)
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=rng.randint(0, 10_000))
        total_reward = 0.0
        t0 = time.time()
        done = False
        
        while not done and env.data.time < seconds:
            # Normalize observation
            x = normalizer.apply(obs[None, :]).astype(np.float32)  # (1, 11)
            
            # Get model prediction
            with torch.no_grad():
                pred = model(torch.tensor(x, dtype=torch.float32))[0].numpy()
            
            # Assemble full action
            action = _assemble_action_torch(section, pred, fsm, env.data, env.model)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if not headless:
                env.render()
        
        dt = time.time() - t0
        results.append({"reward": float(total_reward), "time": dt})
        print(f"Episode {ep+1}: reward={total_reward:.3f}, time={dt:.2f}s")
    
    env.close()
    return results


def play_jax(ckpt_path: str, meta_path: str, episodes: int, seconds: float, seed: int, headless: bool = True):
    """Play with JAX model."""
    import jax
    import jax.numpy as jnp
    from .models_jax import load_eqx_with_template
    from passive_walker.core.controller import FSMStateMachine
    
    # Load metadata first to get model dimensions
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Load model with template
    model = load_eqx_with_template(
        ckpt_path, 
        in_dim=meta["in_dim"], 
        out_dim=meta["out_dim"],
        width=meta.get("hidden", 128),
        depth=meta.get("depth", 2)
    )
    
    section = meta["section"]
    normalizer = Normalizer(
        mean=np.array(meta["normalizer"]["mean"], dtype=np.float32),
        std=np.array(meta["normalizer"]["std"], dtype=np.float32)
    )
    
    env = PassiveWalkerEnv(mode="research", use_gui=not headless)
    fsm = FSMStateMachine()
    
    rng = np.random.RandomState(seed)
    results = []
    
    for ep in range(episodes):
        obs, _ = env.reset(seed=rng.randint(0, 10_000))
        total_reward = 0.0
        t0 = time.time()
        done = False
        
        while not done and env.data.time < seconds:
            # Normalize observation
            x = normalizer.apply(obs[None, :]).astype(np.float32)  # (1, 11)
            
            # Get model prediction
            pred = model(jnp.asarray(x))[0]  # (out_dim,)
            pred_np = np.asarray(pred, dtype=np.float32)
            
            # Assemble full action
            action = _assemble_action_jax(section, pred_np, fsm, env.data, env.model)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if not headless:
                env.render()
        
        dt = time.time() - t0
        results.append({"reward": float(total_reward), "time": dt})
        print(f"Episode {ep+1}: reward={total_reward:.3f}, time={dt:.2f}s")
    
    env.close()
    return results


def main():
    p = argparse.ArgumentParser("BC play")
    p.add_argument("--backend", choices=["torch", "jax"], required=True)
    p.add_argument("--section", choices=["hip", "knees", "both", "both-adv"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seconds", type=float, default=5.0)
    p.set_defaults(gui=False)
    p.add_argument("--gui", dest="gui", action="store_true")
    p.add_argument("--no-gui", dest="gui", action="store_false")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Find metadata file
    ckpt_dir = os.path.dirname(args.ckpt)
    ckpt_name = os.path.basename(args.ckpt)
    stem = ckpt_name.replace('.pt', '').replace('.eqx', '')
    meta_path = os.path.join(ckpt_dir, f"{stem}_meta.json")
    
    if not os.path.exists(meta_path):
        sys.exit(f"Metadata file not found: {meta_path}")

    print(f"[OK] Backend={args.backend}  Section={args.section}  Checkpoint={args.ckpt}")
    print(f"[OK] Episodes={args.episodes}  Seconds={args.seconds}  GUI={args.gui}")

    # Backend presence check and execution
    if args.backend == "torch":
        try: 
            import torch  # noqa
            results = play_torch(args.ckpt, meta_path, args.episodes, args.seconds, args.seed or 0, not args.gui)
        except Exception as e: 
            sys.exit(f"Torch backend error: {e}")
    else:
        try: 
            import jax  # noqa
            results = play_jax(args.ckpt, meta_path, args.episodes, args.seconds, args.seed or 0, not args.gui)
        except Exception as e: 
            sys.exit(f"JAX backend error: {e}")
    
    # Print summary
    rewards = [r["reward"] for r in results]
    times = [r["time"] for r in results]
    print(f"\nSummary:")
    print(f"  Episodes: {len(results)}")
    print(f"  Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean time: {np.mean(times):.2f}s ± {np.std(times):.2f}s")


if __name__ == "__main__":
    main()
