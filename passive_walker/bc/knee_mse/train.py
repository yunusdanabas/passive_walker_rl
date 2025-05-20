# passive_walker/bc/knee_mse/train.py
"""
Train a neural network controller for the knee joint using behavior cloning with MSE loss.

This module implements the training pipeline for a neural network that learns to
mimic a Finite State Machine (FSM) controller for the passive walker's knee joint.
The training uses mean squared error (MSE) loss to minimize the difference between
the neural network's predictions and the FSM controller's actions.

Usage:
    python -m passive_walker.bc.knee_mse.train [--epochs N] [--batch B] [--hidden-size H] [--lr LR] [--gpu] [--plot]
"""

import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from passive_walker.controllers.nn.knee_nn import KneeController
from passive_walker.bc.utils import plot_loss_curve
from . import DATA_DIR, set_device

def train_nn_controller(nn_controller, optimizer, demo_obs, demo_labels, num_epochs, batch_size, plot_loss=False):
    """Train the neural network controller using behavior cloning with MSE loss.
    
    Args:
        nn_controller: The neural network controller to train
        optimizer: Optax optimizer instance
        demo_obs: Demonstration observations
        demo_labels: Demonstration labels (FSM actions)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        plot_loss: Whether to plot the loss curve
        
    Returns:
        tuple: (trained neural network controller, loss history)
    """
    opt_state = optimizer.init(eqx.filter(nn_controller, eqx.is_array))

    @jax.jit
    def loss_fn(model, obs, labels):
        """Compute MSE loss between predictions and labels."""
        preds = jax.vmap(model)(obs)
        return jnp.mean((preds - labels) ** 2)

    @jax.jit
    def update(model, opt_state, obs_batch, label_batch):
        """Perform a single training step."""
        grads = jax.grad(loss_fn)(model, obs_batch, label_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return eqx.apply_updates(model, updates), opt_state

    n = demo_obs.shape[0]
    loss_history = []
    for epoch in range(num_epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            nn_controller, opt_state = update(
                nn_controller,
                opt_state,
                demo_obs[idx],
                demo_labels[idx],
            )
        loss = loss_fn(nn_controller, demo_obs, demo_labels)
        loss_history.append(float(loss))
        print(f"[train] epoch {epoch:02d}  loss={loss:.4f}")

    if plot_loss:
        plot_loss_curve(loss_history, save=str(DATA_DIR / 'loss_histories' / 'knee_mse_training_loss.png'))

    return nn_controller, loss_history

def main():
    """Main training script."""
    p = argparse.ArgumentParser(description="Train NN knee controller (MSE BC)")
    p.add_argument("--epochs",      type=int,   default=100, help="Number of epochs")
    p.add_argument("--batch",       type=int,   default=32,  help="Batch size")
    p.add_argument("--hidden-size", type=int,   default=64,  help="Hidden layer size")
    p.add_argument("--lr",          type=float, default=1e-4,help="Learning rate")
    p.add_argument("--gpu",         action="store_true",    help="Use GPU if available")
    p.add_argument("--plot",        action="store_true",    help="Plot training loss curve")
    args = p.parse_args()

    # Set JAX device BEFORE any other imports
    set_device(args.gpu)

    # Load collected demonstrations
    demos = pickle.load(open(DATA_DIR / "knee_mse_demos.pkl", "rb"))
    demo_obs    = jnp.array(demos["obs"])
    demo_labels = jnp.array(demos["labels"])

    # Initialize model & optimizer
    input_size    = demo_obs.shape[1]
    nn_controller = KneeController(input_size=input_size,
                                hidden_size=args.hidden_size)
    optimizer     = optax.adam(args.lr)

    # Train
    print(f"[train] Starting BC for {demo_obs.shape[0]} samples …")
    nn_controller, loss_history = train_nn_controller(
        nn_controller, optimizer,
        demo_obs, demo_labels,
        num_epochs=args.epochs,
        batch_size=args.batch,
        plot_loss=args.plot,
    )

    # Save final weights
    out_file = DATA_DIR / "knee_mse_controller.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(nn_controller, f)
    print(f"[train] Saved trained controller → {out_file}")

    # Save loss history
    if args.plot:
        loss_file = DATA_DIR / "training_loss_history.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_history, f)
        print(f"[train] Saved loss history → {loss_file}")

if __name__ == "__main__":
    main()
