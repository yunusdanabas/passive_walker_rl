#!/usr/bin/env python3
"""
Model Optimization Pipeline

Unified runner for model optimization (hyperparameters, architecture, training strategies).
Generates organized outputs in results/model_optimization/ with visual summaries.

Usage:
    python analysis_code/run_model_optimization.py \\
        --config passive_walker/bc/pipeline_config.yaml \\
        --components hyperparams architecture \\
        --max-trials 25
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from passive_walker.bc.optimization import (
        HyperparameterOptimizer,
        ArchitectureOptimizer,
        AdvancedTrainer,
        MultiObjectiveOptimizer
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure optimization modules exist in passive_walker/bc/optimization/")
    sys.exit(1)


def create_output_structure(base_dir: Path):
    """Create organized output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"optimization_{timestamp}"
    
    (run_dir / "hyperparameters").mkdir(parents=True, exist_ok=True)
    (run_dir / "architecture").mkdir(parents=True, exist_ok=True)
    (run_dir / "advanced_training").mkdir(parents=True, exist_ok=True)
    (run_dir / "multiobjective").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    
    return run_dir


def run_hyperparameter_optimization(config_path: str, output_dir: Path, 
                                   method: str = "random", max_trials: int = 10):
    """Run hyperparameter optimization."""
    print("\n🔧 Hyperparameter Optimization")
    print("=" * 50)
    
    optimizer = HyperparameterOptimizer(config_path, str(output_dir / "hyperparameters"))
    
    search_space = {
        'learning_rate': [1e-4, 3e-4, 5e-4, 1e-3],
        'batch_size': [512, 1024, 2048],
        'epochs': [30, 50, 75],
        'frame_stack': [1, 2, 3]
    }
    
    results = optimizer.optimize(
        method=method,
        search_space=search_space,
        max_trials=max_trials
    )
    
    print(f"✅ Hyperparameter optimization complete!")
    return results


def run_architecture_search(config_path: str, output_dir: Path, max_trials: int = 10):
    """Run architecture search."""
    print("\n🏗️ Architecture Search")
    print("=" * 50)
    
    optimizer = ArchitectureOptimizer(config_path, str(output_dir / "architecture"))
    
    search_space = {
        'hidden_sizes': [[128, 64], [256, 128], [256, 128, 64], [512, 256]],
        'dropout': [0.0, 0.1, 0.2]
    }
    
    results = optimizer.optimize(
        search_space=search_space,
        max_trials=max_trials
    )
    
    print(f"✅ Architecture search complete!")
    return results


def run_advanced_training(config_path: str, output_dir: Path):
    """Run advanced training strategies."""
    print("\n🎓 Advanced Training Strategies")
    print("=" * 50)
    
    trainer = AdvancedTrainer(config_path, str(output_dir / "advanced_training"))
    
    # Only run curriculum learning (data augmentation not supported)
    results = trainer.curriculum_learning(
        stages=[
            {'epochs': 20, 'data_ratio': 0.3},
            {'epochs': 30, 'data_ratio': 0.7},
            {'epochs': 50, 'data_ratio': 1.0}
        ]
    )
    
    print(f"✅ Advanced training complete!")
    return results


def run_multiobjective_optimization(config_path: str, output_dir: Path, n_candidates: int = 6):
    """Run multi-objective optimization."""
    print("\n🎯 Multi-Objective Optimization")
    print("=" * 50)
    
    optimizer = MultiObjectiveOptimizer(config_path, str(output_dir / "multiobjective"))
    
    results = optimizer.optimize(n_candidates=n_candidates)
    
    print(f"✅ Multi-objective optimization complete!")
    return results


def create_optimization_summary(results: dict, output_dir: Path):
    """Create visual optimization summary."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    fig.suptitle('Model Optimization Summary', fontsize=18, fontweight='bold', y=0.98)
    
    # Hyperparameter results (if available)
    if 'hyperparameters' in results and results['hyperparameters']:
        ax = fig.add_subplot(gs[0, 0])
        hp_results = results['hyperparameters']
        
        if 'best_config' in hp_results:
            best = hp_results['best_config']
            text = f"""
BEST HYPERPARAMETERS

Learning Rate:  {best.get('learning_rate', 'N/A')}
Batch Size:     {best.get('batch_size', 'N/A')}
Epochs:         {best.get('epochs', 'N/A')}
Frame Stack:    {best.get('frame_stack', 'N/A')}

Performance:    {hp_results.get('best_performance', 'N/A')}
            """
            ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.axis('off')
            ax.set_title('Hyperparameter Optimization', fontsize=13, fontweight='bold')
    
    # Architecture results (if available)
    if 'architecture' in results and results['architecture']:
        ax = fig.add_subplot(gs[0, 1])
        arch_results = results['architecture']
        
        if 'best_architecture' in arch_results:
            best = arch_results['best_architecture']
            text = f"""
BEST ARCHITECTURE

Hidden Layers:  {best.get('hidden_sizes', 'N/A')}
Dropout:        {best.get('dropout', 'N/A')}

Performance:    {arch_results.get('best_performance', 'N/A')}
            """
            ax.text(0.1, 0.5, text, fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            ax.axis('off')
            ax.set_title('Architecture Search', fontsize=13, fontweight='bold')
    
    # Summary text
    ax = fig.add_subplot(gs[1, :])
    ax.axis('off')
    
    summary = f"""
    OPTIMIZATION PIPELINE SUMMARY
    
    Timestamp:              {results.get('timestamp', 'N/A')}
    Components Run:         {', '.join(results.get('components_run', []))}
    Total Trials:           {results.get('total_trials', 0)}
    Successful Trials:      {results.get('successful_trials', 0)}
    
    Status: ✅ Optimization complete
    """
    
    ax.text(0.05, 0.5, summary, fontsize=12, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    output_path = output_dir / "figures" / "optimization_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Run model optimization pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to pipeline config YAML')
    parser.add_argument('--components', nargs='+', 
                       choices=['hyperparams', 'architecture', 'advanced', 'multiobjective', 'all'],
                       default=['all'],
                       help='Components to run')
    parser.add_argument('--max-trials', type=int, default=25,
                       help='Maximum trials for search components')
    parser.add_argument('--method', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Search method')
    parser.add_argument('--output-dir', type=str, default='results/model_optimization',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Expand 'all' to individual components
    if 'all' in args.components:
        components = ['hyperparams', 'architecture', 'advanced', 'multiobjective']
    else:
        components = args.components
    
    # Print header
    print("=" * 70)
    print("🚀 MODEL OPTIMIZATION PIPELINE")
    print("=" * 70)
    print(f"\n📂 Config:     {args.config}")
    print(f"🧩 Components: {', '.join(components)}")
    print(f"🎯 Max trials: {args.max_trials}")
    print(f"📁 Output:     {args.output_dir}")
    print()
    
    # Create output structure
    base_dir = Path(args.output_dir)
    run_dir = create_output_structure(base_dir)
    
    print(f"📂 Created: {run_dir}")
    print()
    
    # Results container
    results = {
        'config': args.config,
        'components_run': components,
        'max_trials': args.max_trials,
        'method': args.method,
        'timestamp': datetime.now().isoformat(),
        'total_trials': 0,
        'successful_trials': 0
    }
    
    # Run components
    if 'hyperparams' in components:
        hp_results = run_hyperparameter_optimization(
            args.config, run_dir, args.method, args.max_trials
        )
        results['hyperparameters'] = hp_results
        results['total_trials'] += hp_results.get('total_trials', 0)
        results['successful_trials'] += hp_results.get('successful_trials', 0)
    
    if 'architecture' in components:
        arch_results = run_architecture_search(
            args.config, run_dir, args.max_trials
        )
        results['architecture'] = arch_results
        results['total_trials'] += arch_results.get('total_trials', 0)
        results['successful_trials'] += arch_results.get('successful_trials', 0)
    
    if 'advanced' in components:
        adv_results = run_advanced_training(args.config, run_dir)
        results['advanced_training'] = adv_results
    
    if 'multiobjective' in components:
        mo_results = run_multiobjective_optimization(args.config, run_dir)
        results['multiobjective'] = mo_results
    
    # Create summary
    print("\n" + "=" * 70)
    print("📊 Creating summary...")
    summary_fig = create_optimization_summary(results, run_dir)
    
    # Save metadata
    metadata_path = run_dir / "data" / "optimization_metadata.json"
    with open(metadata_path, 'w') as f:
        # Remove non-serializable objects
        clean_results = {k: v for k, v in results.items() 
                        if isinstance(v, (str, int, float, list, dict))}
        json.dump(clean_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Results: {run_dir}")
    print(f"📊 Summary: {summary_fig}")
    print(f"💾 Metadata: {metadata_path}")
    print(f"\n📈 Total trials: {results['total_trials']}")
    print(f"✅ Successful: {results['successful_trials']}")
    
    # Create symlink
    latest_link = base_dir / "latest_optimization"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)
    
    print(f"🔗 Latest: {latest_link}")
    print("=" * 70)


if __name__ == "__main__":
    main()

