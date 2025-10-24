"""
Model Deployment Infrastructure

Fast inference engine, model registry, and A/B testing framework.
"""

from __future__ import annotations
import os
import json
import time
import pickle
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime


class InferenceEngine:
    """
    Fast inference engine for BC models.
    
    Optimized for low-latency prediction with model quantization.
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu", 
                 quantize: bool = False, optimize: bool = True):
        """
        Initialize inference engine.
        
        Args:
            model: Trained model
            device: Device to run on
            quantize: Whether to quantize model
            optimize: Whether to optimize for inference
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Model optimization
        if optimize:
            self._optimize_model()
        
        if quantize:
            self._quantize_model()
        
        # Performance tracking
        self.inference_times = []
        self.total_inferences = 0
    
    def _optimize_model(self):
        """Optimize model for inference."""
        # Enable inference mode
        self.model.eval()
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _quantize_model(self):
        """Quantize model for faster inference."""
        if hasattr(torch.quantization, 'quantize_dynamic'):
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
    
    def predict(self, inputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Fast prediction with timing.
        
        Args:
            inputs: Input data
            
        Returns:
            Predictions
        """
        start_time = time.time()
        
        # Convert to tensor if needed
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs)
        
        inputs = inputs.to(self.device)
        
        # Inference
        with torch.no_grad():
            if hasattr(self.model, 'forward') and 'hidden' in self.model.forward.__code__.co_varnames:
                # Temporal model
                predictions, _ = self.model(inputs)
            else:
                # MLP model
                predictions = self.model(inputs)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        
        # Track timing
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        return predictions
    
    def predict_batch(self, inputs: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Batch prediction for efficiency.
        
        Args:
            inputs: Batch of input data
            
        Returns:
            Batch of predictions
        """
        return self.predict(inputs)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get inference performance statistics.
        
        Returns:
            Performance statistics
        """
        if not self.inference_times:
            return {"mean_time": 0.0, "total_inferences": 0}
        
        return {
            "mean_time": np.mean(self.inference_times),
            "std_time": np.std(self.inference_times),
            "min_time": np.min(self.inference_times),
            "max_time": np.max(self.inference_times),
            "total_inferences": self.total_inferences,
            "inferences_per_second": 1.0 / np.mean(self.inference_times) if self.inference_times else 0.0
        }
    
    def benchmark(self, input_shape: tuple, num_runs: int = 100) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Shape of input data
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        # Generate random inputs
        inputs = torch.randn(num_runs, *input_shape)
        
        # Warmup
        for _ in range(10):
            self.predict(inputs[0:1])
        
        # Benchmark
        start_time = time.time()
        for i in range(num_runs):
            self.predict(inputs[i:i+1])
        total_time = time.time() - start_time
        
        return {
            "total_time": total_time,
            "mean_time_per_inference": total_time / num_runs,
            "inferences_per_second": num_runs / total_time,
            "num_runs": num_runs
        }


class ModelRegistry:
    """
    Simple model registry for versioning and metadata management.
    
    Tracks model performance, configuration, and deployment status.
    """
    
    def __init__(self, registry_path: str = "experiments/model_registry"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry directory
        """
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        
        self.models = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load existing registry from disk."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                self.models = json.load(f)
    
    def _save_registry(self):
        """Save registry to disk."""
        registry_file = os.path.join(self.registry_path, "registry.json")
        
        with open(registry_file, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def register_model(self, model_id: str, model_path: str, 
                      config: Dict[str, Any], performance: Dict[str, float],
                      description: str = "") -> str:
        """
        Register a new model.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to model file
            config: Model configuration
            performance: Model performance metrics
            description: Model description
            
        Returns:
            Model version ID
        """
        version_id = f"{model_id}_v{len(self.models.get(model_id, {})) + 1}"
        
        model_info = {
            "version_id": version_id,
            "model_path": model_path,
            "config": config,
            "performance": performance,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "status": "registered"
        }
        
        if model_id not in self.models:
            self.models[model_id] = {}
        
        self.models[model_id][version_id] = model_info
        self._save_registry()
        
        print(f"Model registered: {version_id}")
        return version_id
    
    def get_model(self, model_id: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """
        Get model information.
        
        Args:
            model_id: Model identifier
            version: Version to get ("latest" or specific version)
            
        Returns:
            Model information or None
        """
        if model_id not in self.models:
            return None
        
        if version == "latest":
            # Get latest version
            versions = list(self.models[model_id].keys())
            if not versions:
                return None
            version = max(versions, key=lambda v: self.models[model_id][v]["timestamp"])
        
        return self.models[model_id].get(version)
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model IDs
        """
        return list(self.models.keys())
    
    def get_model_versions(self, model_id: str) -> List[str]:
        """
        Get all versions of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of version IDs
        """
        if model_id not in self.models:
            return []
        
        return list(self.models[model_id].keys())
    
    def update_model_status(self, model_id: str, version: str, status: str):
        """
        Update model deployment status.
        
        Args:
            model_id: Model identifier
            version: Model version
            status: New status ("registered", "deployed", "retired")
        """
        if model_id in self.models and version in self.models[model_id]:
            self.models[model_id][version]["status"] = status
            self._save_registry()
    
    def compare_models(self, model_id: str) -> Dict[str, Any]:
        """
        Compare different versions of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Comparison results
        """
        if model_id not in self.models:
            return {}
        
        versions = self.models[model_id]
        comparison = {
            "model_id": model_id,
            "num_versions": len(versions),
            "versions": {}
        }
        
        for version_id, model_info in versions.items():
            comparison["versions"][version_id] = {
                "performance": model_info["performance"],
                "timestamp": model_info["timestamp"],
                "status": model_info["status"]
            }
        
        return comparison


class ABTestingFramework:
    """
    Simple A/B testing framework for model comparison.
    
    Compares different models in production-like conditions.
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize A/B testing framework.
        
        Args:
            registry: Model registry
        """
        self.registry = registry
        self.tests = {}
        self.results = {}
    
    def create_test(self, test_id: str, model_a: str, model_b: str,
                   traffic_split: float = 0.5, duration_hours: int = 24) -> str:
        """
        Create A/B test.
        
        Args:
            test_id: Test identifier
            model_a: First model (control)
            model_b: Second model (treatment)
            traffic_split: Fraction of traffic for model B
            duration_hours: Test duration in hours
            
        Returns:
            Test ID
        """
        test_info = {
            "test_id": test_id,
            "model_a": model_a,
            "model_b": model_b,
            "traffic_split": traffic_split,
            "duration_hours": duration_hours,
            "start_time": datetime.now().isoformat(),
            "status": "active",
            "metrics": {"model_a": [], "model_b": []}
        }
        
        self.tests[test_id] = test_info
        return test_id
    
    def log_metric(self, test_id: str, model_version: str, metric_name: str, 
                  metric_value: float, context: Dict[str, Any] = None):
        """
        Log metric for A/B test.
        
        Args:
            test_id: Test identifier
            model_version: Model version ("model_a" or "model_b")
            metric_name: Metric name
            metric_value: Metric value
            context: Additional context
        """
        if test_id not in self.tests:
            return
        
        metric_entry = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        
        self.tests[test_id]["metrics"][model_version].append(metric_entry)
    
    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get A/B test results.
        
        Args:
            test_id: Test identifier
            
        Returns:
            Test results
        """
        if test_id not in self.tests:
            return {}
        
        test_info = self.tests[test_id]
        results = {
            "test_id": test_id,
            "status": test_info["status"],
            "duration": self._calculate_duration(test_info["start_time"]),
            "model_a_metrics": self._aggregate_metrics(test_info["metrics"]["model_a"]),
            "model_b_metrics": self._aggregate_metrics(test_info["metrics"]["model_b"])
        }
        
        return results
    
    def _calculate_duration(self, start_time: str) -> float:
        """Calculate test duration in hours."""
        start = datetime.fromisoformat(start_time)
        now = datetime.now()
        return (now - start).total_seconds() / 3600
    
    def _aggregate_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics for a model."""
        if not metrics:
            return {}
        
        # Group by metric name
        metric_groups = {}
        for metric in metrics:
            name = metric["metric_name"]
            value = metric["metric_value"]
            
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(value)
        
        # Calculate statistics
        aggregated = {}
        for name, values in metric_groups.items():
            aggregated[f"{name}_mean"] = np.mean(values)
            aggregated[f"{name}_std"] = np.std(values)
            aggregated[f"{name}_count"] = len(values)
        
        return aggregated
    
    def end_test(self, test_id: str):
        """End A/B test."""
        if test_id in self.tests:
            self.tests[test_id]["status"] = "ended"
            self.tests[test_id]["end_time"] = datetime.now().isoformat()


def create_inference_engine(model: nn.Module, **kwargs) -> InferenceEngine:
    """Create inference engine with default settings."""
    return InferenceEngine(model, **kwargs)


def create_model_registry(registry_path: str = "experiments/model_registry") -> ModelRegistry:
    """Create model registry with default settings."""
    return ModelRegistry(registry_path)


def create_ab_testing_framework(registry: ModelRegistry) -> ABTestingFramework:
    """Create A/B testing framework."""
    return ABTestingFramework(registry)
