"""
Ensemble Training CLI

Train ensemble of BC models with bootstrap sampling.
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import torch
from passive_walker.bc.advanced.ensemble import train_ensemble_models, save_ensemble, EnsembleModel
from passive_walker.bc.data.dataset import discover_npzs, split_by_episode, load_xy
from passive_walker.bc.utils import set_seed, pick_device, Normalizer


def main():
    """Train ensemble of BC models."""
    parser = argparse.ArgumentParser("Train BC Ensemble")
    parser.add_argument("--data", type=str, default="experiments/data/fsm_demos", help="Data directory")
    parser.add_argument("--section", type=str, default="both", choices=["hip", "knees", "both"], help="Control section")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "lstm", "gru"], help="Base model type")
    parser.add_argument("--n_models", type=int, default=5, help="Number of models in ensemble")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per model")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for temporal models")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--out", type=str, default="experiments/models/ensemble", help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = pick_device(args.gpu)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    files = discover_npzs(args.data)
    train_files, val_files = split_by_episode(files, val_ratio=0.1)
    
    X_train, y_train = load_xy(train_files, args.section, "act", frame_stack=1)
    X_val, y_val = load_xy(val_files, args.section, "act", frame_stack=1)
    
    print(f"Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples")
    print(f"Input dim: {X_train.shape[1]}, Output dim: {y_train.shape[1]}")
    
    # Normalize inputs
    normalizer = Normalizer(
        mean=np.mean(X_train, axis=0),
        std=np.maximum(np.std(X_train, axis=0), 1e-8)
    )
    X_train_norm = normalizer.encode(X_train)
    X_val_norm = normalizer.encode(X_val)
    
    # Model configuration
    model_config = {
        "type": args.model_type,
        "in_dim": X_train.shape[1],
        "out_dim": y_train.shape[1],
        "learning_rate": args.learning_rate,
    }
    
    if args.model_type == "mlp":
        model_config["hidden_sizes"] = [256, 256]
    else:
        model_config["hidden_size"] = args.hidden_size
        model_config["num_layers"] = 1
        model_config["dropout"] = 0.1
    
    # Train ensemble
    print(f"Training ensemble of {args.n_models} {args.model_type} models...")
    models = train_ensemble_models(
        model_config,
        X_train_norm,
        y_train,
        X_val_norm,
        y_val,
        n_models=args.n_models,
        epochs=args.epochs,
        device=device
    )
    
    # Create ensemble
    ensemble = EnsembleModel(models, voting_strategy="mean")
    
    # Test ensemble
    print("Testing ensemble...")
    test_sample = torch.FloatTensor(X_val_norm[:10]).to(device)
    pred, uncertainty = ensemble.predict_with_uncertainty(test_sample)
    diversity = ensemble.get_diversity_metrics(test_sample)
    
    print(f"Ensemble prediction shape: {pred.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Diversity metrics: {diversity}")
    
    # Save ensemble
    metadata = {
        "model_config": model_config,
        "normalizer": {
            "mean": normalizer.mean.tolist(),
            "std": normalizer.std.tolist()
        },
        "data_info": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "input_dim": X_train.shape[1],
            "output_dim": y_train.shape[1]
        },
        "training_args": vars(args)
    }
    
    save_ensemble(ensemble, args.out, metadata)
    print(f"Ensemble saved to {args.out}")


if __name__ == "__main__":
    main()
