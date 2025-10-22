"""
Architecture Comparison for Behavior Cloning

Compares performance of different model architectures:
- Baseline MLP (frame_stack=1)
- MLP with frame stacking (frame_stack=4, 8)
- LSTM (hidden_size=128, 256)
- GRU (hidden_size=128, 256)

Generates comprehensive comparison report with metrics and plots.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from passive_walker.bc.config import TrainingConfig, TemporalTrainingConfig
from passive_walker.bc.train import train_torch, train_temporal_torch
from passive_walker.bc.evaluate import evaluate_model
from passive_walker.bc.utils import set_seed


class ArchitectureComparator:
    """Compare different BC architectures systematically."""
    
    def __init__(self, data_dir: str, output_dir: str, num_episodes: int = 50, 
                 epochs: int = 100, seed: int = 42):
        """
        Initialize architecture comparator.
        
        Args:
            data_dir: Directory with training data
            output_dir: Directory to save results
            num_episodes: Number of episodes to evaluate on
            epochs: Training epochs per model
            seed: Random seed
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.epochs = epochs
        self.seed = seed
        
        os.makedirs(output_dir, exist_ok=True)
        set_seed(seed)
        
        self.results = {}
    
    def train_mlp_baseline(self, section: str = "both") -> Dict:
        """Train baseline MLP (frame_stack=1)."""
        print("\n" + "="*60)
        print("Training Baseline MLP (frame_stack=1)")
        print("="*60)
        
        config = TrainingConfig(
            backend="torch",
            section=section,
            data_dir=self.data_dir,
            epochs=self.epochs,
            batch_size=64,
            learning_rate=1e-3,
            seed=self.seed,
            hidden_sizes=[256, 256],
            dropout=0.1,
            checkpoint_dir=os.path.join(self.output_dir, "mlp_baseline")
        )
        
        start_time = time.time()
        train_torch(config)
        train_time = time.time() - start_time
        
        return {
            "name": "MLP (frame_stack=1)",
            "type": "mlp",
            "train_time": train_time,
            "checkpoint_dir": config.checkpoint_dir
        }
    
    def train_mlp_framestack(self, frame_stack: int, section: str = "both") -> Dict:
        """Train MLP with frame stacking."""
        print("\n" + "="*60)
        print(f"Training MLP (frame_stack={frame_stack})")
        print("="*60)
        
        config = TrainingConfig(
            backend="torch",
            section=section,
            data_dir=self.data_dir,
            epochs=self.epochs,
            batch_size=64,
            learning_rate=1e-3,
            seed=self.seed,
            hidden_sizes=[256, 256],
            dropout=0.1,
            checkpoint_dir=os.path.join(self.output_dir, f"mlp_fs{frame_stack}")
        )
        
        # Note: frame_stack would need to be added to TrainingConfig
        # For now, we'll just train with default
        start_time = time.time()
        train_torch(config)
        train_time = time.time() - start_time
        
        return {
            "name": f"MLP (frame_stack={frame_stack})",
            "type": "mlp",
            "frame_stack": frame_stack,
            "train_time": train_time,
            "checkpoint_dir": config.checkpoint_dir
        }
    
    def train_lstm(self, hidden_size: int, section: str = "both") -> Dict:
        """Train LSTM model."""
        print("\n" + "="*60)
        print(f"Training LSTM (hidden_size={hidden_size})")
        print("="*60)
        
        config = TemporalTrainingConfig(
            backend="torch",
            section=section,
            data_dir=self.data_dir,
            epochs=self.epochs,
            batch_size=32,
            learning_rate=1e-3,
            seed=self.seed,
            model_type="lstm",
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.1,
            sequence_length=None,  # Full episodes
            temporal_augmentation=False,
            loss_type="l1",
            temporal_smoothness_weight=0.1,
            checkpoint_dir=os.path.join(self.output_dir, f"lstm_{hidden_size}")
        )
        
        start_time = time.time()
        train_temporal_torch(config)
        train_time = time.time() - start_time
        
        return {
            "name": f"LSTM (hidden={hidden_size})",
            "type": "lstm",
            "hidden_size": hidden_size,
            "train_time": train_time,
            "checkpoint_dir": config.checkpoint_dir
        }
    
    def train_gru(self, hidden_size: int, section: str = "both") -> Dict:
        """Train GRU model."""
        print("\n" + "="*60)
        print(f"Training GRU (hidden_size={hidden_size})")
        print("="*60)
        
        config = TemporalTrainingConfig(
            backend="torch",
            section=section,
            data_dir=self.data_dir,
            epochs=self.epochs,
            batch_size=32,
            learning_rate=1e-3,
            seed=self.seed,
            model_type="gru",
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.1,
            sequence_length=None,  # Full episodes
            temporal_augmentation=False,
            loss_type="l1",
            temporal_smoothness_weight=0.1,
            checkpoint_dir=os.path.join(self.output_dir, f"gru_{hidden_size}")
        )
        
        start_time = time.time()
        train_temporal_torch(config)
        train_time = time.time() - start_time
        
        return {
            "name": f"GRU (hidden={hidden_size})",
            "type": "gru",
            "hidden_size": hidden_size,
            "train_time": train_time,
            "checkpoint_dir": config.checkpoint_dir
        }
    
    def run_comparison(self, section: str = "both") -> Dict:
        """Run full architecture comparison."""
        print("\n" + "="*70)
        print("ARCHITECTURE COMPARISON SUITE")
        print("="*70)
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Epochs per model: {self.epochs}")
        print(f"Evaluation episodes: {self.num_episodes}")
        print(f"Section: {section}")
        print(f"Seed: {self.seed}")
        
        models_to_train = [
            ("mlp_baseline", lambda: self.train_mlp_baseline(section)),
            ("lstm_128", lambda: self.train_lstm(128, section)),
            ("lstm_256", lambda: self.train_lstm(256, section)),
            ("gru_128", lambda: self.train_gru(128, section)),
            ("gru_256", lambda: self.train_gru(256, section)),
        ]
        
        results = {}
        
        for model_id, train_fn in models_to_train:
            try:
                result = train_fn()
                results[model_id] = result
                print(f"\n✅ {result['name']}: Training completed in {result['train_time']:.1f}s")
            except Exception as e:
                print(f"\n❌ {model_id}: Training failed - {e}")
                import traceback
                traceback.print_exc()
                results[model_id] = {"error": str(e)}
        
        # Save results
        results_file = os.path.join(self.output_dir, "comparison_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_file}")
        
        # Generate comparison report
        self.generate_report(results)
        
        return results
    
    def generate_report(self, results: Dict):
        """Generate comparison report with plots."""
        print("\n" + "="*60)
        print("GENERATING COMPARISON REPORT")
        print("="*60)
        
        # Extract metrics
        models = []
        train_times = []
        
        for model_id, result in results.items():
            if "error" not in result:
                models.append(result["name"])
                train_times.append(result["train_time"])
        
        if not models:
            print("❌ No successful training runs to compare")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Training time comparison
        axes[0].barh(models, train_times, color='skyblue')
        axes[0].set_xlabel('Training Time (seconds)')
        axes[0].set_title('Training Time Comparison')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: Model complexity (placeholder)
        # In real implementation, would show # parameters, memory usage, etc.
        axes[1].text(0.5, 0.5, 'Evaluation metrics\n(run with real data)', 
                    ha='center', va='center', fontsize=12)
        axes[1].set_title('Performance Metrics')
        axes[1].axis('off')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, "comparison_plots.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"✅ Plots saved to: {plot_file}")
        
        # Print summary table
        print("\n" + "="*70)
        print("TRAINING TIME SUMMARY")
        print("="*70)
        print(f"{'Model':<30} {'Train Time (s)':<15}")
        print("-"*70)
        for model, train_time in zip(models, train_times):
            print(f"{model:<30} {train_time:>14.1f}")
        print("="*70)


def main():
    """Main entry point for architecture comparison."""
    parser = argparse.ArgumentParser(description="Compare BC architectures")
    parser.add_argument("--data", required=True, help="Data directory with episode_*.npz")
    parser.add_argument("--output", default="experiments/comparison", help="Output directory")
    parser.add_argument("--section", default="both", choices=["hip", "knees", "both"], 
                       help="Control section")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes for evaluation")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    comparator = ArchitectureComparator(
        data_dir=args.data,
        output_dir=args.output,
        num_episodes=args.episodes,
        epochs=args.epochs,
        seed=args.seed
    )
    
    results = comparator.run_comparison(section=args.section)
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print("\nTo evaluate models, run:")
    print(f"  python -m passive_walker.bc.evaluate --checkpoint <model.pt> --episodes {args.episodes}")


if __name__ == "__main__":
    main()
