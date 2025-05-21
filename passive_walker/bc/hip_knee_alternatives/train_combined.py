# passive_walker/bc/hip_knee_alternatives/train_combined.py
"""
Train a combined BC model for hip and knee control using MSE loss.

This script trains a neural network to predict both hip and knee actions
using behavioral cloning from FSM demonstrations. The model is trained
to minimize the mean squared error between predicted and demonstrated actions.

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.train_combined [--steps N] [--hz H] [--gpu]
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from passive_walker.bc.hip_knee_alternatives import (
    DATA_BC_HIP_KNEE_ALTERNATIVES,
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
        "--steps", type=int, default=20_000,
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
    args = p.parse_args()

    # Check for existing model with same step count
    model_file = MODEL_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_combined_{args.steps}steps.npz"
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
    model = train_model(obs, labels)

    # save the trained model
    save_model(model, model_file)
    print(f"[train] saved → {model_file}")


if __name__ == "__main__":
    main()
