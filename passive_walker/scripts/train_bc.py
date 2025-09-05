"""
Behavioral Cloning training script.

Trains a neural network policy to imitate FSM expert trajectories.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from passive_walker.core.rollout_buffer import RolloutBuffer


class BCPolicy(nn.Module):
    """Simple MLP policy for behavioral cloning."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build network
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, act_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, obs):
        return self.network(obs)


def load_bc_data(data_dir: str, normalize_obs: bool = True):
    """Load BC data from NPZ files and optionally normalize observations."""
    print(f"Loading BC data from {data_dir}...")

    # Find all NPZ files
    npz_files = glob.glob(os.path.join(data_dir, "episode_*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {data_dir}")

    print(f"Found {len(npz_files)} episode files")

    # Load all episodes
    all_obs = []
    all_acts = []
    all_rews = []
    all_dones = []
    all_info = []
    all_extras = []

    for npz_file in npz_files:
        data = RolloutBuffer.load_npz(npz_file)

        all_obs.append(data["obs"])
        all_acts.append(data["act"])
        all_rews.append(data["rew"])
        all_dones.append(data["done"])
        all_info.append(data["info"])
        if "extras" in data:
            all_extras.append(data["extras"])

    # Concatenate all episodes
    obs = np.concatenate(all_obs, axis=0)
    acts = np.concatenate(all_acts, axis=0)
    rews = np.concatenate(all_rews, axis=0)
    dones = np.concatenate(all_dones, axis=0)
    info = np.concatenate(all_info, axis=0)

    print(f"Loaded {len(obs)} total steps")
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {acts.shape}")
    print(f"Reward range: {rews.min():.3f} to {rews.max():.3f}")

    # Normalize observations if requested
    if normalize_obs:
        obs_mean = obs.mean(axis=0)
        obs_std = obs.std(axis=0)
        obs_std = np.maximum(obs_std, 1e-8)  # Avoid division by zero
        obs = (obs - obs_mean) / obs_std

        norm_stats = {
            "mean": obs_mean,
            "std": obs_std,
        }
        print(f"Normalized observations: mean={obs_mean[:3]}, std={obs_std[:3]}")
    else:
        norm_stats = None

    return obs, acts, rews, dones, info, norm_stats


def train_bc_policy(
    obs, acts, epochs: int, lr: float, batch_size: int, loss_type: str = "mse", device: str = "cpu"
):
    """Train BC policy on expert data."""
    print(f"Training BC policy for {epochs} epochs...")
    print(f"Loss type: {loss_type}, LR: {lr}, Batch size: {batch_size}")

    # Create dataset
    dataset = TensorDataset(torch.FloatTensor(obs), torch.FloatTensor(acts))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = BCPolicy(obs.shape[1], acts.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function
    if loss_type == "mse":
        criterion = nn.MSELoss()
    elif loss_type == "huber":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Training loop
    model.train()
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_obs, batch_acts in dataloader:
            batch_obs = batch_obs.to(device)
            batch_acts = batch_acts.to(device)

            optimizer.zero_grad()
            pred_acts = model(batch_obs)
            loss = criterion(pred_acts, batch_acts)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Loss = {avg_loss:.6f}")

    return model, train_losses


def evaluate_policy(model, obs, acts, device: str = "cpu"):
    """Evaluate policy on test data."""
    model.eval()
    with torch.no_grad():
        pred_acts = model(torch.FloatTensor(obs).to(device))
        pred_acts = pred_acts.cpu().numpy()

    # Compute metrics
    mse = np.mean((pred_acts - acts) ** 2)
    mae = np.mean(np.abs(pred_acts - acts))

    print("Evaluation metrics:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")

    return mse, mae


def main():
    parser = argparse.ArgumentParser(description="Train BC policy on FSM data")
    parser.add_argument(
        "--data_dir", type=str, default="data/bc/raw", help="Directory containing NPZ episode files"
    )
    parser.add_argument(
        "--out_dir", type=str, default="results/bc", help="Output directory for trained policy"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["mse", "huber", "l1"],
        help="Loss function type",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--normalize_obs", action="store_true", help="Normalize observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    obs, acts, rews, dones, info, norm_stats = load_bc_data(
        args.data_dir, normalize_obs=args.normalize_obs
    )

    # Split data (80% train, 20% test)
    n_train = int(0.8 * len(obs))
    train_obs, test_obs = obs[:n_train], obs[n_train:]
    train_acts, test_acts = acts[:n_train], acts[n_train:]

    print(f"Train set: {len(train_obs)} samples")
    print(f"Test set: {len(test_acts)} samples")

    # Train policy
    model, train_losses = train_bc_policy(
        train_obs, train_acts, args.epochs, args.lr, args.batch_size, args.loss, args.device
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_mse, test_mae = evaluate_policy(model, test_obs, test_acts, args.device)

    # Save model and stats
    model_path = os.path.join(args.out_dir, "policy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs.shape[1],
            "act_dim": acts.shape[1],
            "normalize_obs": args.normalize_obs,
            "norm_stats": norm_stats,
            "train_losses": train_losses,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        model_path,
    )

    # Save training summary
    summary = {
        "data_dir": args.data_dir,
        "total_samples": len(obs),
        "train_samples": len(train_obs),
        "test_samples": len(test_obs),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "loss_type": args.loss,
        "normalize_obs": args.normalize_obs,
        "final_train_loss": float(train_losses[-1]),
        "test_mse": float(test_mse),
        "test_mae": float(test_mae),
    }

    summary_path = os.path.join(args.out_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Final test MSE: {test_mse:.6f}")


if __name__ == "__main__":
    main()
