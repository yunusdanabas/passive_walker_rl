"""
Behavioral Cloning Training

Simple BC training for passive walker imitation learning.
"""
from __future__ import annotations
import argparse
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path


# =====================
# Model Parameters
# =====================
HIDDEN_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EPOCHS = 100
EVAL_FREQ = 10


class SimpleMLP(nn.Module):
    """Simple MLP for behavioral cloning."""
    
    def __init__(self, input_dim=11, output_dim=3, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
            nn.Tanh()  # Actions are in [-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)


def load_episodes(data_glob):
    """Load episodes from NPZ files."""
    episodes = []
    files = glob.glob(data_glob)
    
    if not files:
        raise ValueError(f"No files found matching {data_glob}")
    
    print(f"Loading {len(files)} episode files...")
    
    for file_path in files:
        data = np.load(file_path)
        obs = data['obs']  # (T+1, 11)
        act = data['act']  # (T, 3)
        
        # Create state-action pairs
        states = obs[:-1]  # (T, 11)
        actions = act      # (T, 3)
        
        episodes.append((states, actions))
    
    return episodes


def create_datasets(episodes, val_split=0.1):
    """Create train/val datasets from episodes."""
    # Concatenate all episodes
    all_states = np.concatenate([ep[0] for ep in episodes])
    all_actions = np.concatenate([ep[1] for ep in episodes])
    
    # Shuffle
    indices = np.random.permutation(len(all_states))
    all_states = all_states[indices]
    all_actions = all_actions[indices]
    
    # Split
    split_idx = int(len(all_states) * (1 - val_split))
    train_states = all_states[:split_idx]
    train_actions = all_actions[:split_idx]
    val_states = all_states[split_idx:]
    val_actions = all_actions[split_idx:]
    
    return (train_states, train_actions), (val_states, val_actions)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for states, actions in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            
            outputs = model(states)
            loss = criterion(outputs, actions)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser("BC training")
    parser.add_argument("--data", type=str, required=True, help="Data glob pattern")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create output directory
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    
    # Load data
    episodes = load_episodes(args.data)
    (train_states, train_actions), (val_states, val_actions) = create_datasets(episodes)
    
    print(f"Training data: {len(train_states)} samples")
    print(f"Validation data: {len(val_states)} samples")
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_actions)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_states),
        torch.FloatTensor(val_actions)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    print(f"Training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if epoch % EVAL_FREQ == 0 or epoch == args.epochs - 1:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.out:
                    torch.save(model.state_dict(), os.path.join(args.out, "best_model.pth"))
        else:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.6f}")
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    
    # Save final model
    if args.out:
        torch.save(model.state_dict(), os.path.join(args.out, "final_model.pth"))
        print(f"Models saved to {args.out}")


if __name__ == "__main__":
    main()