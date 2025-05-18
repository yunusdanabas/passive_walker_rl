# passive_walker/bc/hip_knee_alternatives/train_combined.py
"""
Train an MLP that outputs [hip, kneeL, kneeR] using a combined loss function.

The loss combines:
- Robust losses (MSE + L1 + Huber)
- Symmetry constraints (hip and knee)
- Smoothness across time
- Energy regularization

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.train_combined [--epochs N] [--batch B]
                                                                   [--hidden-size H] [--lr LR]
                                                                   [--alpha-sym A] [--beta-smooth B]
                                                                   [--gamma-energy G] [--gpu] [--plot]
"""

import argparse, pickle, numpy as np, jax, jax.numpy as jnp, optax, equinox as eqx
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.plotters import plot_loss_curve
from . import DATA_DIR, set_device


def train_nn_controller(model, optimizer, obs, labels, epochs, batch, alpha_sym=0.1, 
                       beta_smooth=0.1, gamma_energy=0.01, plot_loss=False):
    """Return (trained_model, loss_history)."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_hist = []

    @jax.jit
    def loss_fn(m, o, y):
        preds = jax.vmap(m)(o)  # shape: (N, 3)
        
        # 1. Robust losses
        mse_loss = jnp.mean(jnp.square(preds - y))
        l1_loss = jnp.mean(jnp.abs(preds - y))
        huber_loss = jnp.mean(optax.huber_loss(preds - y, delta=1.0))
        robust_loss = (mse_loss + l1_loss + huber_loss) / 3.0
        
        # 2. Symmetry losses
        # Knee symmetry: difference between left and right knee commands
        knee_sym_loss = jnp.mean((preds[:, 1] - preds[:, 2]) ** 2)
        # Hip symmetry: enforce hip output to be close to mean of labels
        hip_target = jnp.mean(y[:, 0])
        hip_sym_loss = jnp.mean((preds[:, 0] - hip_target) ** 2)
        symmetry_loss = hip_sym_loss + knee_sym_loss
        
        # 3. Smoothness loss: differences between consecutive outputs
        smoothness_loss = jnp.mean(jnp.sum((preds[1:] - preds[:-1]) ** 2, axis=1))
        
        # 4. Energy loss: penalize high magnitude actions
        energy_loss = jnp.mean(jnp.sum(preds ** 2, axis=1))
        
        # Combine all losses
        total_loss = (robust_loss + 
                     alpha_sym * symmetry_loss + 
                     beta_smooth * smoothness_loss + 
                     gamma_energy * energy_loss)
        
        return total_loss

    @jax.jit
    def step(m, st, o, y):
        g = jax.grad(loss_fn)(m, o, y)
        upd, st = optimizer.update(g, st)
        return eqx.apply_updates(m, upd), st

    n = obs.shape[0]
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            model, opt_state = step(model, opt_state, obs[idx], labels[idx])
        loss = float(loss_fn(model, obs, labels))
        loss_hist.append(loss)
        print(f"[COMBINED] epoch {ep:02d}  loss={loss:.4f}")

    if plot_loss:
        plot_loss_curve(loss_hist, save=str(DATA_DIR / 'loss_histories' / 'hip_knee_combined_training_loss.png'))

    return model, loss_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch",        type=int,   default=32)
    p.add_argument("--hidden-size",  type=int,   default=128)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--alpha-sym",    type=float, default=0.1)
    p.add_argument("--beta-smooth",  type=float, default=0.1)
    p.add_argument("--gamma-energy", type=float, default=0.01)
    p.add_argument("--gpu",          action="store_true")
    p.add_argument("--plot",         action="store_true")
    args = p.parse_args()

    set_device(args.gpu)

    demos = pickle.load(open(DATA_DIR / "hip_knee_alternatives_demos.pkl", "rb"))
    obs, labels = jnp.array(demos["obs"]), jnp.array(demos["labels"])

    model = HipKneeController(
        input_size=obs.shape[1],
        hidden_size=args.hidden_size,
        key=jax.random.PRNGKey(42)
    )
    optimizer = optax.adam(args.lr)

    model, loss_hist = train_nn_controller(
        model, optimizer, obs, labels,
        epochs=args.epochs, batch=args.batch,
        alpha_sym=args.alpha_sym,
        beta_smooth=args.beta_smooth,
        gamma_energy=args.gamma_energy,
        plot_loss=args.plot
    )

    out = DATA_DIR / "hip_knee_combined_controller.pkl"
    pickle.dump(model, open(out, "wb"))
    print(f"[train] saved → {out}")

    if args.plot:
        loss_file = DATA_DIR / "training_loss_history.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_hist, f)
        print(f"[train] saved loss history → {loss_file}")


if __name__ == "__main__":
    main()
