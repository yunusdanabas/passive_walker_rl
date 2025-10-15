"""
BC Training Pipeline

Simple YAML-based pipeline runner that uses existing BC functions.
Handles data collection, training, evaluation, and basic plotting.

Usage:
    python -m passive_walker.bc.run_pipeline
    python -m passive_walker.bc.run_pipeline --config configs/pipeline_config.yaml
    python -m passive_walker.bc.run_pipeline --preset quick_test
"""

from __future__ import annotations
import argparse
import os
import sys
import yaml
import subprocess
import time
from pathlib import Path

# Import existing BC functions
from .utils import set_seed, ensure_dir
from .dataset import discover_npzs


def load_config(config_path: str, preset: str = None) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply preset if specified
    if preset and preset in config:
        preset_config = config[preset].copy()
        # Merge with base config, preset overrides base
        base_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
        config = {**base_config, **preset_config}
    
    return config


def run_pipeline(config: dict):
    """Run complete BC pipeline using existing functions."""
    print("\n" + "="*60)
    print("BC TRAINING PIPELINE")
    print("="*60)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Create results directory
    ensure_dir(config['results_dir'])
    
    print(f"[PIPELINE] Section: {config['section']}, Backend: {config['backend']}")
    print(f"[PIPELINE] Episodes: {config['episodes']}, Epochs: {config['epochs']}")
    
    try:
        # Step 1: Collect data if needed
        collect_data_if_needed(config)
        
        # Step 2: Train model
        train_model(config)
        
        # Step 3: Evaluate model
        evaluate_model(config)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)


def collect_data_if_needed(config: dict):
    """Collect FSM data if not available."""
    print(f"\n--- STEP 1: DATA COLLECTION ---")
    
    # Check if data already exists
    try:
        files = discover_npzs(config['data_dir'])
        if len(files) >= config['min_episodes']:
            print(f"[INFO] Found {len(files)} episodes in {config['data_dir']}")
            return
    except Exception:
        pass
    
    print(f"[INFO] Collecting {config['episodes']} FSM episodes...")
    
    # Use existing FSM collect function
    cmd = [
        "python", "-m", "passive_walker.fsm.collect",
        "--episodes", str(config['episodes']),
        "--steps", str(config['steps']),
        "--out", config['data_dir'],
        "--seed", str(config['seed'])
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Data collection failed: {result.stderr}")
    
    print(f"[INFO] Data collection completed")


def train_model(config: dict):
    """Train BC model using existing train function."""
    print(f"\n--- STEP 2: MODEL TRAINING ---")
    
    # Use existing BC train function
    cmd = [
        "python", "-m", "passive_walker.bc.train",
        "--backend", config['backend'],
        "--section", config['section'],
        "--data", config['data_dir'],
        "--label-type", config['label_type'],
        "--epochs", str(config['epochs']),
        "--batch", str(config['batch_size']),
        "--lr", str(config['learning_rate']),
        "--seed", str(config['seed']),
        "--save-dir", config['results_dir']
    ]
    
    # Add advanced loss weights if using both-adv
    if config['section'] == "both-adv":
        loss_weights = config['loss_weights']
        cmd.extend([
            "--w1", str(loss_weights['w1']),
            "--w2", str(loss_weights['w2']),
            "--w3", str(loss_weights['w3']),
            "--w4", str(loss_weights['w4'])
        ])
    
    # Add frame stacking if specified
    if config.get('frame_stack', 1) > 1:
        cmd.extend(["--frame-stack", str(config['frame_stack'])])
    
    print(f"[CMD] {' '.join(cmd)}")
    
    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    training_time = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"Training failed: {result.stderr}")
    
    print(f"[INFO] Training completed in {training_time:.1f}s")


def evaluate_model(config: dict):
    """Evaluate trained model using existing play function."""
    print(f"\n--- STEP 3: MODEL EVALUATION ---")
    
    # Find the trained model files
    model_path, meta_path = find_model_files(config)
    
    # Use existing BC play function
    cmd = [
        "python", "-m", "passive_walker.bc.play",
        "--ckpt", model_path,
        "--meta", meta_path,
        "--episodes", str(config['eval_episodes']),
        "--seconds", str(config['eval_seconds']),
        "--seed", str(config['seed'] + 1000)  # Different seed for evaluation
    ]
    
    # Add frame stacking if specified
    if config.get('frame_stack', 1) > 1:
        cmd.extend(["--frame-stack", str(config['frame_stack'])])
    
    if not config.get('gui', True):
        cmd.append("--no-gui")
    
    print(f"[CMD] {' '.join(cmd)}")
    
    # Run evaluation
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    eval_time = time.time() - start_time
    
    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")
    
    print(f"[INFO] Evaluation completed in {eval_time:.1f}s")
    print(f"[INFO] Evaluation output:")
    print(result.stdout)


def find_model_files(config: dict):
    """Find the trained model and metadata files."""
    results_dir = Path(config['results_dir'])
    
    # Look for model files
    if config['backend'] == "torch":
        model_ext = ".pt"
    else:
        model_ext = ".eqx"
    
    model_files = list(results_dir.glob(f"*{model_ext}"))
    meta_files = list(results_dir.glob("*_meta.json"))
    
    if not model_files:
        raise FileNotFoundError(f"No {model_ext} files found in {results_dir}")
    
    # Use the most recent model file
    model_path = max(model_files, key=os.path.getmtime)
    
    # Find corresponding metadata file
    model_stem = model_path.stem
    meta_path = None
    for meta_file in meta_files:
        if model_stem in meta_file.stem:
            meta_path = meta_file
            break
    
    if not meta_path:
        raise FileNotFoundError(f"No metadata file found for {model_path}")
    
    return str(model_path), str(meta_path)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser("BC Training Pipeline")
    parser.add_argument("--config", default="passive_walker/bc/pipeline_config.yaml",
                       help="Path to YAML configuration file")
    parser.add_argument("--preset", choices=["quick_test", "full_control", "advanced", "advanced_hip"],
                       help="Use a preset configuration")
    
    # Direct parameter overrides
    parser.add_argument("--section", choices=["hip", "knees", "both", "both-adv"],
                       help="Control section")
    parser.add_argument("--backend", choices=["torch", "jax"], help="Training backend")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--steps", type=int, help="Steps per episode")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--frame-stack", type=int, help="Frame stacking")
    parser.add_argument("--eval-episodes", type=int, help="Evaluation episodes")
    parser.add_argument("--eval-seconds", type=float, help="Evaluation seconds")
    parser.add_argument("--gui", action="store_true", help="Enable GUI")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Disable GUI")
    parser.add_argument("--results-dir", help="Results directory")
    parser.add_argument("--data-dir", help="Data directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.preset)
    
    # Override with CLI arguments if provided
    if args.section:
        config['section'] = args.section
    if args.backend:
        config['backend'] = args.backend
    if args.episodes:
        config['episodes'] = args.episodes
    if args.steps:
        config['steps'] = args.steps
    if args.epochs:
        config['epochs'] = args.epochs
    if args.batch:
        config['batch_size'] = args.batch
    if args.lr:
        config['learning_rate'] = args.lr
    if args.frame_stack:
        config['frame_stack'] = args.frame_stack
    if args.eval_episodes:
        config['eval_episodes'] = args.eval_episodes
    if args.eval_seconds:
        config['eval_seconds'] = args.eval_seconds
    if args.gui is not None:
        config['gui'] = args.gui
    if args.results_dir:
        config['results_dir'] = args.results_dir
    if args.data_dir:
        config['data_dir'] = args.data_dir
    
    # Run pipeline
    run_pipeline(config)


if __name__ == "__main__":
    main()