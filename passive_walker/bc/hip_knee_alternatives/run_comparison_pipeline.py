# passive_walker/bc/hip_knee_alternatives/run_comparison_pipeline.py
"""
Run and compare all hip+knee BC variants (MSE, Huber, L1, Combined), plot losses,
evaluate each in Mujoco GUI, and play out the best-performing controller.

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.run_comparison_pipeline \
        [--steps N] [--epochs E] [--batch B] [--hidden-size H] \
        [--lr LR] [--sim-duration S] [--gpu]
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import optax
from mujoco.glfw import glfw

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from .collect import collect_demo_data
from . import DATA_DIR, XML_PATH, set_device

# import each training variant
import passive_walker.bc.hip_knee_alternatives.train_mse    as mse_mod
import passive_walker.bc.hip_knee_alternatives.train_huber  as huber_mod
import passive_walker.bc.hip_knee_alternatives.train_l1     as l1_mod
import passive_walker.bc.hip_knee_alternatives.train_combined as comb_mod

def train_variant(name, train_fn, obs, labels, args, **extra_kwargs):
    """Train one variant and return (model, loss_history)."""
    print(f"\n=== Training {name.upper()} variant ===")
    # init model & optimizer
    model = HipKneeController(input_size=obs.shape[1],
                              hidden_size=args.hidden_size,
                              key=jax.random.PRNGKey(args.seed))
    optimizer = optax.adam(args.lr)
    # call appropriate train function
    if name == "combined":
        model, loss_hist = train_fn(
            model, optimizer, obs, labels,
            epochs=args.epochs,
            batch=args.batch,
            alpha_sym=args.alpha_sym,
            beta_smooth=args.beta_smooth,
            gamma_energy=args.gamma_energy,
            weight_decay=args.weight_decay
        )
    else:
        # mse, huber, l1 all define train(model, optimizer, obs, labels, epochs, batch)
        model, loss_hist = train_fn(
            model, optimizer, obs, labels,
            epochs=args.epochs,
            batch=args.batch
        )

    # save model and loss history
    model_file = DATA_DIR / f"controller_{name}.pkl"
    hist_file  = DATA_DIR / f"loss_{name}.pkl"
    pickle.dump(model, open(model_file, "wb"))
    pickle.dump(loss_hist, open(hist_file, "wb"))
    print(f"[{name}] saved model → {model_file}")
    return model, loss_hist

def rollout(model, sim_duration, use_gpu):
    """Run one Mujoco GUI rollout and return total reward."""
    # configure device and imports
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True
    )
    obs = env.reset()
    done = False
    total = 0.0
    while not done and not glfw.window_should_close(env.window):
        act = np.array(model(jnp.array(obs)))
        obs, r, done, _ = env.step(act)
        total += r
        env.render()
    env.close()
    return total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps",        type=int,   default=20_000, help="Demo steps")
    p.add_argument("--epochs",       type=int,   default=50,    help="Training epochs")
    p.add_argument("--batch",        type=int,   default=32,    help="Batch size")
    p.add_argument("--hidden-size",  type=int,   default=128,   help="Hidden layer size")
    p.add_argument("--lr",           type=float, default=1e-4,  help="Learning rate")
    p.add_argument("--sim-duration", type=float, default=30.0,  help="GUI simulation duration (s)")
    p.add_argument("--seed",         type=int,   default=42,    help="PRNG seed")
    p.add_argument("--alpha_sym",    type=float, default=0.1,   help="Symmetry weight (combined)")
    p.add_argument("--beta_smooth",  type=float, default=0.1,   help="Smoothness weight (combined)")
    p.add_argument("--gamma_energy", type=float, default=0.01,  help="Energy weight (combined)")
    p.add_argument("--weight_decay", type=float, default=1e-4,  help="Weight decay (combined)")
    p.add_argument("--gpu",          action="store_true",       help="Use GPU if available")
    args = p.parse_args()

    # configure JAX backend
    set_device(args.gpu)

    # 1) collect demos
    print("\n=== Step 1: Collecting demos ===")
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.steps / 60.0,
        use_nn_for_hip=False,
        use_nn_for_knees=False,
        use_gui=False
    )
    obs, labels = collect_demo_data(env, num_steps=args.steps)
    env.close()
    # convert to JAX arrays
    obs_j = jnp.array(obs)
    lab_j = jnp.array(labels)
    print(f"Collected {obs_j.shape[0]} samples → obs={obs_j.shape}, labels={lab_j.shape}")

    # 2) train each variant
    variants = {
        "mse":      mse_mod.train,
        "huber":    huber_mod.train,
        "l1":       l1_mod.train,
        "combined": comb_mod.train_combined,
    }
    results = {}
    for name, fn in variants.items():
        mdl, hist = train_variant(name, fn, obs_j, lab_j, args)
        results[name] = {"model": mdl, "loss": hist}

    # 3) plot all loss curves
    plt.figure(figsize=(8,5))
    for name, res in results.items():
        plt.plot(res["loss"], label=name.upper())
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("BC Loss Comparison")
    plt.legend()
    out_fig = DATA_DIR / "loss_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved loss comparison plot → {out_fig}")

    # 4) evaluate each variant in GUI (short rollout)
    print("\n=== Step 4: GUI rollouts ===")
    rewards = {}
    for name, res in results.items():
        print(f"\n{name.upper()} rollout:")
        total = rollout(res["model"], args.sim_duration, args.gpu)
        print(f"  total reward = {total:.3f}")
        rewards[name] = total

    # 5) select best and play longer
    best = max(rewards, key=rewards.get)
    print(f"\nBest variant: {best.upper()} (reward {rewards[best]:.3f})")
    print("Playing best variant for full sim…")
    # final long rollout
    final_total = rollout(results[best]["model"], args.sim_duration*2, args.gpu)
    print(f"\nFinal long rollout reward = {final_total:.3f}")

if __name__ == "__main__":
    main()
