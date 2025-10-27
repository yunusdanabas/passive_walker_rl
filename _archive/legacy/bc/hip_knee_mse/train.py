"""
Train an MLP that outputs [hip, kneeL, kneeR] using MSE behaviour cloning.

This module implements the training pipeline for a neural network that learns to
mimic a Finite State Machine (FSM) controller for the passive walker's hip and knee joints.
The training uses mean squared error (MSE) loss to minimize the difference between
the neural network's predictions and the FSM controller's actions.

Usage:
    python -m passive_walker.bc.hip_knee_mse.train [--epochs N] [--batch B]
                                                  [--hidden-size H] [--lr LR]
                                                  [--steps N] [--gpu] [--plot]
"""

import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.utils import plot_loss_curve
from passive_walker.bc.hip_knee_mse import DATA_BC_HIP_KNEE_MSE, RESULTS_BC_HIP_KNEE_MSE, set_device, save_model


def train_nn_controller(model, optimizer, obs, labels, epochs, batch, plot_loss=True, steps=None):
    """
    Train the neural network controller using behavior cloning with MSE loss.
    
    Args:
        model: The neural network controller to train
        optimizer: Optax optimizer instance
        obs: Demonstration observations
        labels: Demonstration labels (FSM actions)
        epochs: Number of training epochs
        batch: Batch size for training
        plot_loss: Whether to plot the loss curve
        steps: Number of steps in demo data for file naming
        
    Returns:
        tuple: (trained neural network controller, loss history)
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_hist = []

    @jax.jit
    def loss_fn(m, o, y):
        return jnp.mean((jax.vmap(m)(o) - y) ** 2)

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
        print(f"[train] epoch {ep:02d}  loss={loss:.4f}")

    if plot_loss:
        plot_loss_curve(loss_hist, save=str(RESULTS_BC_HIP_KNEE_MSE / 'loss_histories' / f'hip_knee_mse_training_loss_{steps}steps.png'))

    return model, loss_hist


def main():
    """Main training script."""
    p = argparse.ArgumentParser(description="Train NN hip+knee controller (MSE BC)")
    p.add_argument("--epochs",      type=int,   default=50, help="Number of epochs")
    p.add_argument("--batch",       type=int,   default=32,  help="Batch size")
    p.add_argument("--hidden-size", type=int,   default=256, help="Hidden layer size")
    p.add_argument("--lr",          type=float, default=3e-4,help="Learning rate")
    p.add_argument("--steps",       type=int,   default=50_000, help="Number of steps in demo data")
    p.add_argument("--gpu",         action="store_true",    help="Use GPU if available")
    p.add_argument("--plot",        action="store_true",    help="Plot training loss curve")
    args = p.parse_args()

    # Set JAX device BEFORE any other imports
    set_device(args.gpu)

    # Load collected demonstrations with step count in filename
    demo_file = DATA_BC_HIP_KNEE_MSE / f"hip_knee_mse_demos_{args.steps}steps.pkl"
    if not demo_file.exists():
        raise FileNotFoundError(f"No demo file found for {args.steps} steps. Please run collect.py first.")
    
    demos = pickle.load(open(demo_file, "rb"))
    obs, labels = jnp.array(demos["obs"]), jnp.array(demos["labels"])

    # Initialize model & optimizer
    model = HipKneeController(input_size=obs.shape[1], hidden_size=args.hidden_size)
    optimizer = optax.adam(args.lr)

    # Train
    print(f"[train] Starting BC for {obs.shape[0]} samples from {args.steps} steps demo…")
    model, loss_hist = train_nn_controller(
        model, optimizer, obs, labels,
        epochs=args.epochs, batch=args.batch,
        plot_loss=args.plot, steps=args.steps
    )

    # Save final weights with step count in filename
    out_file = RESULTS_BC_HIP_KNEE_MSE / f"hip_knee_mse_controller_{args.steps}steps.eqx"
    save_model(model, out_file)
    print(f"[train] Saved trained controller → {out_file}")

    # Save loss history with step count in filename
    if args.plot:
        loss_file = RESULTS_BC_HIP_KNEE_MSE / f"training_loss_history_{args.steps}steps.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_hist, f)
        print(f"[train] Saved loss history → {loss_file}")


if __name__ == "__main__":
    main()
