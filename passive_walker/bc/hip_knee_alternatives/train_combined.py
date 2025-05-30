# passive_walker/bc/hip_knee_alternatives/train_combined.py
"""
Train a combined BC model for hip and knee control using combined loss.

This script trains a neural network to predict both hip and knee actions
using behavioral cloning from FSM demonstrations. The model is trained
to minimize the mean squared error between predicted and demonstrated actions.

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.train_combined [--steps N] [--hz H] [--gpu]
"""

import argparse
import pickle

from passive_walker.bc.hip_knee_alternatives import (
    DATA_BC_HIP_KNEE_ALTERNATIVES,
    RESULTS_BC_HIP_KNEE_ALTERNATIVES,
    MODEL_BC_HIP_KNEE_ALTERNATIVES,
    load_demo_data,
    set_device,
    train_model,
    save_model,
)


def main():
    """Main function to train the combined BC model."""
    p = argparse.ArgumentParser(
        description="Train combined BC model for hip+knee control"
    )
    p.add_argument(
        "--steps", type=int, default=50_000,
        help="Number of simulation steps to use for training"
    )
    p.add_argument(
        "--hz", type=int, default=200,
        help="Simulation frequency in Hz"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available (sets JAX_PLATFORM_NAME=gpu)"
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Save loss history to file"
    )
    args = p.parse_args()

    # Check for existing model with same step count
    model_file = MODEL_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_combined_{args.steps}steps.eqx"
    if model_file.exists():
        print(f"[train] Found existing model with {args.steps} steps → {model_file}")
        return

    # configure JAX backend
    set_device(args.gpu)

    # load demonstration data
    demo_file = DATA_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_demos_{args.steps}steps.pkl"
    if not demo_file.exists():
        print(f"[train] No demo file found with {args.steps} steps → {demo_file}")
        return

    print(f"[train] loading demos from {demo_file}")
    obs, labels = load_demo_data(demo_file)

    # train the model
    print(f"[train] training on {len(obs)} samples…")
    model, loss_history = train_model(obs, labels)

    # Save model
    out_file = RESULTS_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_combined_controller_{args.steps}steps.eqx"
    save_model(model, out_file)
    print(f"[train] Saved trained controller → {out_file}")

    # Save loss history
    if args.plot:
        loss_file = RESULTS_BC_HIP_KNEE_ALTERNATIVES / f"training_loss_history_combined_{args.steps}steps.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_history, f)
        print(f"[train] Saved loss history → {loss_file}")


if __name__ == "__main__":
    main()
