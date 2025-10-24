"""
Ensemble Methods for Behavior Cloning

Simple ensemble training and inference with voting strategies.
Supports multiple base models (MLP, LSTM, GRU) with diversity metrics.
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
from passive_walker.bc.models.models_torch import TorchMLP
from passive_walker.bc.models.temporal_torch import TorchLSTM, TorchGRU, TorchBiLSTM
from passive_walker.bc.utils import Normalizer


class EnsembleModel:
    """
    Ensemble of BC models with voting strategies.
    
    Supports mean, weighted mean, and uncertainty-weighted voting.
    """
    
    def __init__(self, models: List[nn.Module], voting_strategy: str = "mean"):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            voting_strategy: "mean", "weighted_mean", or "uncertainty_weighted"
        """
        self.models = models
        self.voting_strategy = voting_strategy
        self.device = next(models[0].parameters()).device
        
        # Move all models to same device
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble prediction.
        
        Args:
            x: Input tensor (batch, features) or (batch, seq_len, features)
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                    # Temporal model - use single step inference
                    pred, _ = model(x)
                else:
                    # MLP model
                    pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch, out_dim)
        
        if self.voting_strategy == "mean":
            return predictions.mean(dim=0)
        elif self.voting_strategy == "weighted_mean":
            # Equal weights for now - can be extended
            weights = torch.ones(len(self.models), device=self.device) / len(self.models)
            weights = weights.view(-1, 1, 1)  # (n_models, 1, 1)
            return (predictions * weights).sum(dim=0)
        elif self.voting_strategy == "uncertainty_weighted":
            # Weight by inverse variance
            mean_pred = predictions.mean(dim=0)
            variance = predictions.var(dim=0)
            weights = 1.0 / (variance + 1e-8)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return (predictions * weights.unsqueeze(0)).sum(dim=0)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty estimate.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                    pred, _ = model(x)
                else:
                    pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch, out_dim)
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty
    
    def get_diversity_metrics(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of diversity metrics
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                if hasattr(model, 'forward') and 'hidden' in model.forward.__code__.co_varnames:
                    pred, _ = model(x)
                else:
                    pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch, out_dim)
        
        # Mean pairwise disagreement
        disagreement = 0.0
        n_pairs = 0
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                disagreement += torch.mean(torch.abs(predictions[i] - predictions[j])).item()
                n_pairs += 1
        
        disagreement /= n_pairs if n_pairs > 0 else 1
        
        # Prediction variance
        variance = torch.mean(predictions.var(dim=0)).item()
        
        return {
            "disagreement": disagreement,
            "prediction_variance": variance,
            "n_models": len(self.models)
        }


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str], 
    model_configs: List[Dict[str, Any]],
    voting_strategy: str = "mean"
) -> EnsembleModel:
    """
    Create ensemble from saved checkpoints.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        model_configs: List of model configurations
        voting_strategy: Voting strategy to use
        
    Returns:
        Trained ensemble model
    """
    models = []
    
    for checkpoint_path, config in zip(checkpoint_paths, model_configs):
        # Create model based on config
        if config["type"] == "mlp":
            model = TorchMLP(
                in_dim=config["in_dim"],
                out_dim=config["out_dim"],
                hidden=config["hidden_sizes"][0]  # Use first hidden size
            )
        elif config["type"] == "lstm":
            model = TorchLSTM(
                in_dim=config["in_dim"],
                out_dim=config["out_dim"],
                hidden_size=config["hidden_size"],
                num_layers=config.get("num_layers", 1),
                dropout=config.get("dropout", 0.1)
            )
        elif config["type"] == "gru":
            model = TorchGRU(
                in_dim=config["in_dim"],
                out_dim=config["out_dim"],
                hidden_size=config["hidden_size"],
                num_layers=config.get("num_layers", 1),
                dropout=config.get("dropout", 0.1)
            )
        elif config["type"] == "bilstm":
            model = TorchBiLSTM(
                in_dim=config["in_dim"],
                out_dim=config["out_dim"],
                hidden_size=config["hidden_size"],
                num_layers=config.get("num_layers", 1),
                dropout=config.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        models.append(model)
    
    return EnsembleModel(models, voting_strategy)


def bootstrap_sample_data(X: np.ndarray, y: np.ndarray, n_samples: int) -> List[tuple[np.ndarray, np.ndarray]]:
    """
    Create bootstrap samples for ensemble training.
    
    Args:
        X: Input data
        y: Target data
        n_samples: Number of bootstrap samples
        
    Returns:
        List of (X_sample, y_sample) tuples
    """
    n_data = len(X)
    samples = []
    
    for _ in range(n_samples):
        # Sample with replacement
        indices = np.random.choice(n_data, size=n_data, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        samples.append((X_sample, y_sample))
    
    return samples


def train_ensemble_models(
    base_config: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_models: int = 5,
    epochs: int = 50,
    device: str = "cpu"
) -> List[nn.Module]:
    """
    Train ensemble of models with bootstrap sampling.
    
    Args:
        base_config: Base model configuration
        X_train: Training inputs
        y_train: Training targets
        X_val: Validation inputs
        X_val: Validation targets
        n_models: Number of models in ensemble
        epochs: Training epochs per model
        device: Device to train on
        
    Returns:
        List of trained models
    """
    models = []
    
    for i in range(n_models):
        print(f"Training ensemble model {i+1}/{n_models}")
        
        # Bootstrap sample
        X_sample, y_sample = bootstrap_sample_data(X_train, y_train, 1)[0]
        
        # Create model
        if base_config["type"] == "mlp":
            model = TorchMLP(
                in_dim=base_config["in_dim"],
                out_dim=base_config["out_dim"],
                hidden=base_config["hidden_sizes"][0]  # Use first hidden size
            )
        elif base_config["type"] == "lstm":
            model = TorchLSTM(
                in_dim=base_config["in_dim"],
                out_dim=base_config["out_dim"],
                hidden_size=base_config["hidden_size"],
                num_layers=base_config.get("num_layers", 1),
                dropout=base_config.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"Unsupported model type for ensemble: {base_config['type']}")
        
        model.to(device)
        
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=base_config.get("learning_rate", 1e-3))
        criterion = torch.nn.L1Loss()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_sample).to(device)
        y_tensor = torch.FloatTensor(y_sample).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            if base_config["type"] in ["lstm", "gru", "bilstm"]:
                # Temporal model - add sequence dimension
                X_seq = X_tensor.unsqueeze(1)  # (batch, 1, features)
                pred, _ = model(X_seq)
                pred = pred.squeeze(1)  # Remove sequence dimension
            else:
                pred = model(X_tensor)
            
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            # Validation
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    if base_config["type"] in ["lstm", "gru", "bilstm"]:
                        X_val_seq = X_val_tensor.unsqueeze(1)
                        val_pred, _ = model(X_val_seq)
                        val_pred = val_pred.squeeze(1)
                    else:
                        val_pred = model(X_val_tensor)
                    
                    val_loss = criterion(val_pred, y_val_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict().copy()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        models.append(model)
        print(f"Model {i+1} trained, best val loss: {best_val_loss:.4f}")
    
    return models


def save_ensemble(ensemble: EnsembleModel, save_dir: str, metadata: Dict[str, Any]):
    """
    Save ensemble model and metadata.
    
    Args:
        ensemble: Trained ensemble
        save_dir: Directory to save to
        metadata: Model metadata
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual models
    for i, model in enumerate(ensemble.models):
        model_path = os.path.join(save_dir, f"model_{i}.pt")
        torch.save(model.state_dict(), model_path)
    
    # Save metadata
    metadata["voting_strategy"] = ensemble.voting_strategy
    metadata["n_models"] = len(ensemble.models)
    
    metadata_path = os.path.join(save_dir, "ensemble_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Ensemble saved to {save_dir}")


def load_ensemble(load_dir: str) -> EnsembleModel:
    """
    Load ensemble from saved directory.
    
    Args:
        load_dir: Directory containing ensemble files
        
    Returns:
        Loaded ensemble model
    """
    # Load metadata
    metadata_path = os.path.join(load_dir, "ensemble_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load models
    models = []
    for i in range(metadata["n_models"]):
        model_path = os.path.join(load_dir, f"model_{i}.pt")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model based on metadata
        if metadata["type"] == "mlp":
            model = TorchMLP(
                in_dim=metadata["in_dim"],
                out_dim=metadata["out_dim"],
                hidden=metadata["hidden_sizes"][0]  # Use first hidden size
            )
        elif metadata["type"] == "lstm":
            model = TorchLSTM(
                in_dim=metadata["in_dim"],
                out_dim=metadata["out_dim"],
                hidden_size=metadata["hidden_size"],
                num_layers=metadata.get("num_layers", 1),
                dropout=metadata.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {metadata['type']}")
        
        model.load_state_dict(checkpoint)
        models.append(model)
    
    return EnsembleModel(models, metadata["voting_strategy"])
