"""
Experiment Configuration and Management

Simple experiment configuration system for BC training.
Manages hyperparameters, model configs, and experiment metadata.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from passive_walker.bc.experiment.tracking import ExperimentTracker


@dataclass
class ExperimentConfig:
    """Configuration for BC training experiment."""
    
    # Experiment metadata
    experiment_name: str
    description: str = ""
    tags: List[str] = None
    
    # Model configuration
    model_type: str = "mlp"  # "mlp", "lstm", "gru", "bilstm"
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.1
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    optimizer: str = "adam"  # "adam", "sgd", "rmsprop"
    scheduler: str = "none"  # "none", "plateau", "cosine"
    
    # Data configuration
    data_dir: str = "experiments/data/fsm_demos"
    section: str = "both"  # "hip", "knees", "both"
    validation_split: float = 0.2
    
    # Advanced features
    use_ensemble: bool = False
    ensemble_size: int = 5
    use_uncertainty: bool = False
    use_curriculum: bool = False
    use_multitask: bool = False
    use_augmentation: bool = False
    
    # Environment configuration
    ctrl_hz: int = 100
    randomization_profile: str = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.tags is None:
            self.tags = []
        
        # Validate model type
        if self.model_type not in ["mlp", "lstm", "gru", "bilstm"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        # Validate optimizer
        if self.optimizer not in ["adam", "sgd", "rmsprop"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        
        # Validate scheduler
        if self.scheduler not in ["none", "plateau", "cosine"]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        
        # Validate section
        if self.section not in ["hip", "knees", "both"]:
            raise ValueError(f"Invalid section: {self.section}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class ExperimentRunner:
    """
    Simple experiment runner for BC training.
    
    Manages experiment lifecycle: setup, training, evaluation, cleanup.
    """
    
    def __init__(self, config: ExperimentConfig, log_dir: str = "experiments/logs"):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            log_dir: Directory for experiment logs
        """
        self.config = config
        self.log_dir = log_dir
        
        # Initialize tracking
        self.tracker = ExperimentTracker(log_dir, config.experiment_name)
        self.tracker.set_hyperparameters(config.to_dict())
        
        # Experiment state
        self.start_time = None
        self.end_time = None
        self.results = {}
    
    def setup(self):
        """Setup experiment (validate config, create directories)."""
        print(f"Setting up experiment: {self.config.experiment_name}")
        
        # Validate data directory
        if not os.path.exists(self.config.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
        
        # Create output directories
        output_dir = os.path.join(self.log_dir, self.config.experiment_name, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        self.config.save(config_path)
        
        print("✓ Experiment setup complete")
    
    def run_training(self, train_fn, **kwargs):
        """
        Run training with experiment tracking.
        
        Args:
            train_fn: Training function
            **kwargs: Additional arguments for training function
        """
        print(f"Starting training: {self.config.experiment_name}")
        self.start_time = time.time()
        
        # Add tracker to training arguments
        kwargs["tracker"] = self.tracker
        
        try:
            # Run training
            results = train_fn(self.config, **kwargs)
            self.results.update(results)
            
            print("✓ Training completed successfully")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
        
        finally:
            self.end_time = time.time()
    
    def run_evaluation(self, eval_fn, **kwargs):
        """
        Run evaluation with experiment tracking.
        
        Args:
            eval_fn: Evaluation function
            **kwargs: Additional arguments for evaluation function
        """
        print(f"Running evaluation: {self.config.experiment_name}")
        
        try:
            # Run evaluation
            eval_results = eval_fn(self.config, **kwargs)
            self.results.update(eval_results)
            
            print("✓ Evaluation completed successfully")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise
    
    def finalize(self):
        """Finalize experiment (log final metrics, save results)."""
        print(f"Finalizing experiment: {self.config.experiment_name}")
        
        # Calculate experiment duration
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            self.results["experiment_duration"] = duration
            print(f"Experiment duration: {duration:.2f} seconds")
        
        # Log final hyperparameters and metrics
        self.tracker.log_hyperparameters(self.results)
        
        # Save experiment info
        experiment_info = {
            "config": self.config.to_dict(),
            "results": self.results,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time and self.start_time else None
        }
        
        self.tracker.save_experiment_info(experiment_info)
        
        # Close tracker
        self.tracker.close()
        
        print("✓ Experiment finalized")


def create_experiment_config(experiment_name: str, **kwargs) -> ExperimentConfig:
    """
    Create experiment configuration with default values.
    
    Args:
        experiment_name: Name of experiment
        **kwargs: Configuration overrides
        
    Returns:
        Experiment configuration
    """
    return ExperimentConfig(experiment_name=experiment_name, **kwargs)


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """
    Create default experiment configurations.
    
    Returns:
        Dictionary of default configurations
    """
    configs = {}
    
    # Basic MLP experiment
    configs["mlp_basic"] = ExperimentConfig(
        experiment_name="mlp_basic",
        description="Basic MLP model for BC",
        model_type="mlp",
        hidden_size=128,
        epochs=50
    )
    
    # LSTM experiment
    configs["lstm_temporal"] = ExperimentConfig(
        experiment_name="lstm_temporal",
        description="LSTM model for temporal BC",
        model_type="lstm",
        hidden_size=128,
        num_layers=2,
        epochs=100
    )
    
    # Ensemble experiment
    configs["ensemble_advanced"] = ExperimentConfig(
        experiment_name="ensemble_advanced",
        description="Ensemble of MLP models",
        model_type="mlp",
        use_ensemble=True,
        ensemble_size=5,
        epochs=50
    )
    
    # Multi-task experiment
    configs["multitask_learning"] = ExperimentConfig(
        experiment_name="multitask_learning",
        description="Multi-task learning with auxiliary tasks",
        model_type="lstm",
        use_multitask=True,
        use_curriculum=True,
        epochs=100
    )
    
    # Advanced experiment
    configs["advanced_features"] = ExperimentConfig(
        experiment_name="advanced_features",
        description="All advanced features enabled",
        model_type="lstm",
        use_ensemble=True,
        use_uncertainty=True,
        use_curriculum=True,
        use_multitask=True,
        use_augmentation=True,
        epochs=150
    )
    
    return configs


def compare_experiments(experiment_names: List[str], log_dir: str = "experiments/logs"):
    """
    Compare multiple experiments.
    
    Args:
        experiment_names: List of experiment names to compare
        log_dir: Directory containing experiment logs
    """
    print(f"Comparing experiments: {experiment_names}")
    
    results = {}
    
    for exp_name in experiment_names:
        exp_dir = os.path.join(log_dir, exp_name)
        info_path = os.path.join(exp_dir, "experiment_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                exp_data = json.load(f)
            
            results[exp_name] = {
                "config": exp_data.get("hyperparameters", {}),
                "metrics": exp_data.get("final_metrics", {}),
                "duration": exp_data.get("duration", None)
            }
        else:
            print(f"⚠️  Experiment info not found: {exp_name}")
    
    # Print comparison
    print("\n📊 Experiment Comparison:")
    print("-" * 80)
    
    for exp_name, data in results.items():
        print(f"\n🔬 {exp_name}:")
        print(f"  Duration: {data['duration']:.2f}s" if data['duration'] else "  Duration: N/A")
        
        metrics = data['metrics']
        if metrics:
            print("  Final Metrics:")
            for metric_name, metric_value in metrics.items():
                print(f"    {metric_name}: {metric_value:.4f}")
    
    return results
