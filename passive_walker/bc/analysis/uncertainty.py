"""
Uncertainty Estimation for Behavior Cloning

Monte Carlo Dropout and ensemble-based uncertainty quantification.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
from passive_walker.bc.advanced.ensemble import EnsembleModel


class MCDropoutModel(nn.Module):
    """
    Model with Monte Carlo Dropout for uncertainty estimation.
    
    Wraps existing models to enable dropout during inference.
    """
    
    def __init__(self, base_model: nn.Module, dropout_rate: float = 0.1):
        """
        Initialize MC Dropout model.
        
        Args:
            base_model: Base model to wrap
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        
        # Enable dropout layers
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout in all dropout layers."""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during inference
    
    def forward(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MC Dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        predictions = []
        
        for _ in range(n_samples):
            if hasattr(self.base_model, 'forward') and 'hidden' in self.base_model.forward.__code__.co_varnames:
                # Temporal model
                pred, _ = self.base_model(x)
            else:
                # MLP model
                pred = self.base_model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch, out_dim)
        
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_pred, uncertainty


class UncertaintyEstimator:
    """
    Unified uncertainty estimation interface.
    
    Supports both MC Dropout and ensemble-based uncertainty.
    """
    
    def __init__(self, model: Union[nn.Module, EnsembleModel], method: str = "ensemble"):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: Model or ensemble to use
            method: "mc_dropout" or "ensemble"
        """
        self.model = model
        self.method = method
        
        if method == "mc_dropout":
            if not isinstance(model, nn.Module):
                raise ValueError("MC Dropout requires a single model")
            self.uncertainty_model = MCDropoutModel(model)
        elif method == "ensemble":
            if not isinstance(model, EnsembleModel):
                raise ValueError("Ensemble method requires EnsembleModel")
            self.uncertainty_model = model
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty estimate.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for MC Dropout
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        if self.method == "mc_dropout":
            return self.uncertainty_model(x, n_samples)
        elif self.method == "ensemble":
            return self.uncertainty_model.predict_with_uncertainty(x)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_uncertainty_metrics(self, x: torch.Tensor, n_samples: int = 10) -> Dict[str, float]:
        """
        Compute uncertainty metrics.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for MC Dropout
            
        Returns:
            Dictionary of uncertainty metrics
        """
        pred, uncertainty = self.predict_with_uncertainty(x, n_samples)
        
        # Mean uncertainty across batch and outputs
        mean_uncertainty = torch.mean(uncertainty).item()
        
        # Max uncertainty
        max_uncertainty = torch.max(uncertainty).item()
        
        # Uncertainty coefficient of variation
        uncertainty_std = torch.std(uncertainty).item()
        uncertainty_cv = uncertainty_std / (mean_uncertainty + 1e-8)
        
        return {
            "mean_uncertainty": mean_uncertainty,
            "max_uncertainty": max_uncertainty,
            "uncertainty_cv": uncertainty_cv,
            "method": self.method
        }


def create_mc_dropout_model(base_model: nn.Module, dropout_rate: float = 0.1) -> MCDropoutModel:
    """
    Create MC Dropout wrapper for existing model.
    
    Args:
        base_model: Base model to wrap
        dropout_rate: Dropout probability
        
    Returns:
        MC Dropout model
    """
    return MCDropoutModel(base_model, dropout_rate)


def evaluate_uncertainty_calibration(
    estimator: UncertaintyEstimator,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_samples: int = 10
) -> Dict[str, float]:
    """
    Evaluate uncertainty calibration.
    
    Args:
        estimator: Uncertainty estimator
        X_test: Test inputs
        y_test: Test targets
        n_samples: Number of MC samples
        
    Returns:
        Calibration metrics
    """
    pred, uncertainty = estimator.predict_with_uncertainty(X_test, n_samples)
    
    # Compute prediction errors
    errors = torch.abs(pred - y_test)
    
    # Uncertainty should correlate with errors
    errors_flat = errors.flatten()
    uncertainty_flat = uncertainty.flatten()
    
    # Correlation between uncertainty and error
    correlation = torch.corrcoef(torch.stack([errors_flat, uncertainty_flat]))[0, 1].item()
    
    # Calibration: check if high uncertainty regions have high errors
    uncertainty_threshold = torch.quantile(uncertainty_flat, 0.8)
    high_uncertainty_mask = uncertainty_flat > uncertainty_threshold
    
    if high_uncertainty_mask.sum() > 0:
        high_uncertainty_errors = errors_flat[high_uncertainty_mask].mean().item()
        low_uncertainty_errors = errors_flat[~high_uncertainty_mask].mean().item()
        calibration_ratio = high_uncertainty_errors / (low_uncertainty_errors + 1e-8)
    else:
        calibration_ratio = 1.0
    
    return {
        "uncertainty_error_correlation": correlation,
        "calibration_ratio": calibration_ratio,
        "mean_error": errors.mean().item(),
        "mean_uncertainty": uncertainty.mean().item()
    }


def visualize_uncertainty(
    estimator: UncertaintyEstimator,
    X_sample: torch.Tensor,
    y_sample: torch.Tensor,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create uncertainty visualization.
    
    Args:
        estimator: Uncertainty estimator
        X_sample: Sample inputs
        y_sample: Sample targets
        save_path: Optional path to save plot
        
    Returns:
        Visualization data
    """
    import matplotlib.pyplot as plt
    
    pred, uncertainty = estimator.predict_with_uncertainty(X_sample)
    
    # Convert to numpy for plotting
    pred_np = pred.detach().cpu().numpy()
    uncertainty_np = uncertainty.detach().cpu().numpy()
    y_np = y_sample.detach().cpu().numpy()
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prediction vs target
    axes[0, 0].scatter(y_np.flatten(), pred_np.flatten(), alpha=0.6)
    axes[0, 0].plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'r--')
    axes[0, 0].set_xlabel('Target')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].set_title('Prediction vs Target')
    
    # Error vs uncertainty
    errors = np.abs(pred_np - y_np)
    axes[0, 1].scatter(uncertainty_np.flatten(), errors.flatten(), alpha=0.6)
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Absolute Error')
    axes[0, 1].set_title('Error vs Uncertainty')
    
    # Uncertainty distribution
    axes[1, 0].hist(uncertainty_np.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Uncertainty Distribution')
    
    # Prediction confidence intervals
    std_uncertainty = np.sqrt(uncertainty_np)
    axes[1, 1].errorbar(range(len(pred_np)), pred_np[:, 0], yerr=std_uncertainty[:, 0], 
                       fmt='o', alpha=0.6, capsize=3)
    axes[1, 1].plot(range(len(y_np)), y_np[:, 0], 'r-', alpha=0.8, label='Target')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Prediction')
    axes[1, 1].set_title('Predictions with Uncertainty')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Uncertainty visualization saved to {save_path}")
    
    plt.show()
    
    return {
        "prediction": pred_np,
        "uncertainty": uncertainty_np,
        "targets": y_np,
        "errors": errors
    }
