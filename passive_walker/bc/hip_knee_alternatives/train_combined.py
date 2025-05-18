# passive_walker/bc/hip_knee_alternatives/train_combined.py
"""
Train an MLP for [hip, left_knee, right_knee] with a composite loss:

    total = imitation(MSE)
          + α·knee_symmetry
          + β·smoothness
          + γ·energy
          + δ·weight_decay

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.train_combined \
        [--epochs N] [--batch B] [--hidden-size H] [--lr LR] \
        [--alpha_sym A] [--beta_smooth B] [--gamma_energy G] \
        [--weight-decay D] [--gpu]
"""

import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.plotters import plot_loss_curve
from . import DATA_DIR, set_device


def train_combined(
    model: HipKneeController,
    optimizer: optax.GradientTransformation,
    obs: jnp.ndarray,
    labels: jnp.ndarray,
    epochs: int,
    batch: int,
    alpha_sym: float,
    beta_smooth: float,
    gamma_energy: float,
    weight_decay: float
) -> tuple[HipKneeController, list[float]]:
    """
    Train with composite loss:
      • imitation_loss = MSE(preds, labels)
      • knee_symmetry  = MSE((preds[:,1] - preds[:,2]) - (labels[:,1] - labels[:,2]), 0)
      • smoothness     = MSE(preds[t+1] - preds[t], 0)
      • energy         = mean(sum(preds**2, axis=1))
      • weight_decay   = sum(params**2)
    """
    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_hist = []

    @jax.jit
    def loss_fn(m, o, y):
        preds = jax.vmap(m)(o)  # shape (N,3)

        # 1) imitation MSE
        imitation = jnp.mean((preds - y) ** 2)

        # 2) label-derived knee symmetry
        true_diff = y[:, 1] - y[:, 2]
        pred_diff = preds[:, 1] - preds[:, 2]
        knee_sym  = jnp.mean((pred_diff - true_diff) ** 2)

        # 3) smoothness across time
        diffs  = preds[1:] - preds[:-1]
        smooth = jnp.mean(jnp.sum(diffs ** 2, axis=1))

        # 4) energy penalty
        energy = jnp.mean(jnp.sum(preds ** 2, axis=1))

        # 5) weight decay on parameters
        l2 = 0.0
        for p in eqx.filter(model, eqx.is_array):
            l2 = l2 + jnp.sum(p ** 2)

        return (
            imitation
            + alpha_sym   * knee_sym
            + beta_smooth * smooth
            + gamma_energy* energy
            + weight_decay * l2
        )

    @jax.jit
    def step(m, st, o, y):
        grads = jax.grad(loss_fn)(m, o, y)
        updates, st = optimizer.update(grads, st)
        m = eqx.apply_updates(m, updates)
        return m, st

    n = obs.shape[0]
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            model, opt_state = step(model, opt_state, obs[idx], labels[idx])
        current = float(loss_fn(model, obs, labels))
        loss_hist.append(current)
        print(f"[combined] epoch {ep:02d}  total_loss={current:.4f}")

    return model, loss_hist


def main():
    p = argparse.ArgumentParser(description="Train hip+knee MLP with combined loss")
    p.add_argument("--epochs",        type=int,   default=100,   help="Training epochs")
    p.add_argument("--batch",         type=int,   default=32,    help="Batch size")
    p.add_argument("--hidden-size",   type=int,   default=128,   help="Hidden layer size")
    p.add_argument("--lr",            type=float, default=1e-4,  help="Learning rate")
    p.add_argument("--alpha_sym",     type=float, default=0.1,   help="Weight for knee symmetry")
    p.add_argument("--beta_smooth",   type=float, default=0.1,   help="Weight for smoothness")
    p.add_argument("--gamma_energy",  type=float, default=0.01,  help="Weight for energy penalty")
    p.add_argument("--weight-decay",  type=float, default=1e-4,  help="L2 parameter regularization")
    p.add_argument("--gpu",           action="store_true",      help="Use GPU if available")
    args = p.parse_args()

    # Configure JAX backend
    set_device(args.gpu)

    # Load collected demos
    demos = pickle.load(open(DATA_DIR / "hip_knee_alternatives_demos.pkl", "rb"))
    obs, labels = jnp.array(demos["obs"]), jnp.array(demos["labels"])

    # Initialize model and optimizer
    model     = HipKneeController(input_size=obs.shape[1], hidden_size=args.hidden_size)
    optimizer = optax.adam(args.lr)

    # Train with combined loss
    print(f"[combined] training on {obs.shape[0]} samples…")
    model, loss_hist = train_combined(
        model,
        optimizer,
        obs,
        labels,
        epochs=args.epochs,
        batch=args.batch,
        alpha_sym=args.alpha_sym,
        beta_smooth=args.beta_smooth,
        gamma_energy=args.gamma_energy,
        weight_decay=args.weight_decay,
    )

    # Plot training loss curve
    plot_loss_curve(loss_hist)

    # Save final model
    out = DATA_DIR / "controller_combined.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"[combined] saved → {out}")


if __name__ == "__main__":
    main()
