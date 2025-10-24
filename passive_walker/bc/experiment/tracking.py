"""
TensorBoard Experiment Tracking

Simple experiment tracking with TensorBoard for BC training.
Logs metrics, hyperparameters, and model outputs.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, Optional, Union
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    """
    Simple TensorBoard experiment tracker.
    
    Logs scalars, distributions, images, and hyperparameters.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Initialize experiment tracker.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        if experiment_name is None:
            experiment_name = f"exp_{int(time.time())}"
        
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        # Track hyperparameters
        self.hparams = {}
        self.metrics = {}
        
        print(f"Experiment tracker initialized: {self.log_dir}")
    
    def log_scalar(self, name: str, value: float, step: int):
        """
        Log scalar metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        self.writer.add_scalar(name, value, step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int):
        """
        Log multiple scalar metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_distribution(self, name: str, values: Union[np.ndarray, torch.Tensor], step: int):
        """
        Log distribution histogram.
        
        Args:
            name: Distribution name
            values: Values to histogram
            step: Training step
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        self.writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image: Union[np.ndarray, torch.Tensor], step: int):
        """
        Log image.
        
        Args:
            name: Image name
            image: Image data
            step: Training step
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        self.writer.add_image(name, image, step)
    
    def log_text(self, name: str, text: str, step: int):
        """
        Log text.
        
        Args:
            name: Text name
            text: Text content
            step: Training step
        """
        self.writer.add_text(name, text, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """
        Log model computational graph.
        
        Args:
            model: PyTorch model
            input_sample: Sample input for graph tracing
        """
        self.writer.add_graph(model, input_sample)
    
    def set_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Set hyperparameters for logging.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        self.hparams = hparams.copy()
    
    def log_hyperparameters(self, metrics: Dict[str, float]):
        """
        Log hyperparameters and final metrics.
        
        Args:
            metrics: Final evaluation metrics
        """
        self.metrics = metrics.copy()
        
        # Convert non-serializable values to strings
        hparams_str = {}
        for key, value in self.hparams.items():
            if isinstance(value, (list, tuple)):
                hparams_str[key] = str(value)
            elif isinstance(value, dict):
                hparams_str[key] = str(value)
            else:
                hparams_str[key] = value
        
        self.writer.add_hparams(hparams_str, metrics)
    
    def log_training_step(self, step: int, losses: Dict[str, float], 
                         learning_rate: float = None, grad_norm: float = None):
        """
        Log training step metrics.
        
        Args:
            step: Training step
            losses: Dictionary of loss values
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        # Log losses
        for loss_name, loss_value in losses.items():
            self.log_scalar(f"train/{loss_name}", loss_value, step)
        
        # Log learning rate
        if learning_rate is not None:
            self.log_scalar("train/learning_rate", learning_rate, step)
        
        # Log gradient norm
        if grad_norm is not None:
            self.log_scalar("train/grad_norm", grad_norm, step)
    
    def log_validation_step(self, step: int, metrics: Dict[str, float]):
        """
        Log validation step metrics.
        
        Args:
            step: Training step
            metrics: Dictionary of validation metrics
        """
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f"val/{metric_name}", metric_value, step)
    
    def log_model_predictions(self, step: int, predictions: torch.Tensor, 
                             targets: torch.Tensor, name: str = "predictions"):
        """
        Log model predictions and targets.
        
        Args:
            step: Training step
            predictions: Model predictions
            targets: Ground truth targets
            name: Name for logging
        """
        # Log prediction distributions
        self.log_distribution(f"{name}/predictions", predictions, step)
        self.log_distribution(f"{name}/targets", targets, step)
        
        # Log prediction errors
        errors = torch.abs(predictions - targets)
        self.log_distribution(f"{name}/errors", errors, step)
        
        # Log mean absolute error
        mae = torch.mean(errors).item()
        self.log_scalar(f"{name}/mae", mae, step)
    
    def log_uncertainty_metrics(self, step: int, uncertainty: torch.Tensor, 
                               errors: torch.Tensor, name: str = "uncertainty"):
        """
        Log uncertainty estimation metrics.
        
        Args:
            step: Training step
            uncertainty: Uncertainty estimates
            errors: Prediction errors
            name: Name for logging
        """
        # Log uncertainty distribution
        self.log_distribution(f"{name}/uncertainty", uncertainty, step)
        
        # Log uncertainty metrics
        mean_uncertainty = torch.mean(uncertainty).item()
        max_uncertainty = torch.max(uncertainty).item()
        
        self.log_scalar(f"{name}/mean", mean_uncertainty, step)
        self.log_scalar(f"{name}/max", max_uncertainty, step)
        
        # Log uncertainty-error correlation
        if len(uncertainty) > 1:
            uncertainty_flat = uncertainty.flatten()
            errors_flat = errors.flatten()
            
            correlation = torch.corrcoef(torch.stack([uncertainty_flat, errors_flat]))[0, 1].item()
            self.log_scalar(f"{name}/error_correlation", correlation, step)
    
    def save_experiment_info(self, info: Dict[str, Any]):
        """
        Save experiment information to JSON file.
        
        Args:
            info: Experiment information
        """
        info_path = os.path.join(self.log_dir, "experiment_info.json")
        
        experiment_data = {
            "experiment_name": self.experiment_name,
            "timestamp": time.time(),
            "hyperparameters": self.hparams,
            "final_metrics": self.metrics,
            "info": info
        }
        
        with open(info_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Experiment info saved to {info_path}")
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
        print(f"Experiment tracker closed: {self.experiment_name}")


class ExperimentManager:
    """
    Simple experiment manager for organizing multiple experiments.
    """
    
    def __init__(self, base_log_dir: str = "experiments/logs"):
        """
        Initialize experiment manager.
        
        Args:
            base_log_dir: Base directory for all experiments
        """
        self.base_log_dir = base_log_dir
        os.makedirs(base_log_dir, exist_ok=True)
        
        self.experiments = {}
    
    def create_experiment(self, experiment_name: str, **kwargs) -> ExperimentTracker:
        """
        Create new experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            **kwargs: Additional arguments for ExperimentTracker
            
        Returns:
            Experiment tracker instance
        """
        tracker = ExperimentTracker(self.base_log_dir, experiment_name, **kwargs)
        self.experiments[experiment_name] = tracker
        return tracker
    
    def get_experiment(self, experiment_name: str) -> Optional[ExperimentTracker]:
        """
        Get existing experiment tracker.
        
        Args:
            experiment_name: Name of experiment
            
        Returns:
            Experiment tracker or None
        """
        return self.experiments.get(experiment_name)
    
    def list_experiments(self) -> list[str]:
        """
        List all experiment names.
        
        Returns:
            List of experiment names
        """
        return list(self.experiments.keys())
    
    def close_all(self):
        """Close all experiment trackers."""
        for tracker in self.experiments.values():
            tracker.close()
        self.experiments.clear()


def create_experiment_tracker(log_dir: str = "experiments/logs", 
                            experiment_name: str = None) -> ExperimentTracker:
    """
    Create experiment tracker with default settings.
    
    Args:
        log_dir: Directory for logs
        experiment_name: Name of experiment
        
    Returns:
        Experiment tracker
    """
    return ExperimentTracker(log_dir, experiment_name)


def log_training_metrics(tracker: ExperimentTracker, step: int, 
                        train_losses: Dict[str, float],
                        val_losses: Dict[str, float] = None,
                        learning_rate: float = None,
                        grad_norm: float = None):
    """
    Log comprehensive training metrics.
    
    Args:
        tracker: Experiment tracker
        step: Training step
        train_losses: Training losses
        val_losses: Validation losses
        learning_rate: Current learning rate
        grad_norm: Gradient norm
    """
    # Log training metrics
    tracker.log_training_step(step, train_losses, learning_rate, grad_norm)
    
    # Log validation metrics
    if val_losses is not None:
        tracker.log_validation_step(step, val_losses)
    
    # Log combined metrics
    all_metrics = {"train": train_losses}
    if val_losses is not None:
        all_metrics["val"] = val_losses
    
    for split, metrics in all_metrics.items():
        for metric_name, metric_value in metrics.items():
            tracker.log_scalar(f"{split}/{metric_name}", metric_value, step)
