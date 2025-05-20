"""
Train an MLP that outputs [hip, kneeL, kneeR] using MSE behaviour cloning.

Usage:
    python -m passive_walker.bc.hip_knee_mse.train [--epochs N] [--batch B]
                                                  [--hidden-size H] [--lr LR]
                                                  [--gpu] [--plot]
"""

import argparse, pickle, numpy as np, jax, jax.numpy as jnp, optax, equinox as eqx
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.utils                import plot_loss_curve
from . import DATA_DIR, set_device


def train_nn_controller(model, optimizer, obs, labels, epochs, batch, plot_loss=False):
    """Return (trained_model, loss_history)."""
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_hist = []

    @jax.jit
    def loss_fn(m, o, y):
        return jnp.mean((jax.vmap(m)(o) - y) ** 2)

    @jax.jit
    def step(m, st, o, y):
        g   = jax.grad(loss_fn)(m, o, y)
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
        print(f"[train] epoch {ep:02d}  loss={loss:.4f}")

    if plot_loss:
        plot_loss_curve(loss_hist, save=str(DATA_DIR / 'loss_histories' / 'hip_knee_mse_training_loss.png'))

    return model, loss_hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--hidden-size", type=int,   default=128)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--gpu",         action="store_true")
    p.add_argument("--plot",        action="store_true")
    args = p.parse_args()

    set_device(args.gpu)

    demos       = pickle.load(open(DATA_DIR / "hip_knee_mse_demos.pkl", "rb"))
    obs, labels = jnp.array(demos["obs"]), jnp.array(demos["labels"])

    model     = HipKneeController(input_size=obs.shape[1], hidden_size=args.hidden_size)
    optimizer = optax.adam(args.lr)

    model, loss_hist = train_nn_controller(
        model, optimizer, obs, labels,
        epochs=args.epochs, batch=args.batch,
        plot_loss=args.plot
    )

    out = DATA_DIR / "hip_knee_mse_controller.pkl"
    pickle.dump(model, open(out, "wb"))
    print(f"[train] saved → {out}")

    if args.plot:
        loss_file = DATA_DIR / "training_loss_history.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_hist, f)
        print(f"[train] saved loss history → {loss_file}")


if __name__ == "__main__":
    main()
