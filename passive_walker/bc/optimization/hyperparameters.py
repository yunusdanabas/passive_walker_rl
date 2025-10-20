"""
Hyperparameter Optimizer for Phase 3

Implements grid search and random search optimization for training hyperparameters.
Uses analysis results from Phase 1 & 2 to guide search space.
"""

import os
import json
import itertools
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import subprocess
import sys


class HyperparameterOptimizer:
    """Optimizes training hyperparameters using grid/random search."""
    
    def __init__(self, base_config_path: str, results_dir: str = "results/optimization"):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        with open(base_config_path, 'r') as f:
            self.base_config = self._load_config(base_config_path)
        
        # Search spaces based on common effective ranges
        self.search_spaces = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
            'batch_size': [256, 512, 1024, 2048],
            'epochs': [20, 30, 40, 50, 75, 100],
            'frame_stack': [1, 2, 3, 4],
            'loss_weights': {
                'w1': [0.5, 1.0, 1.5],  # L1 loss weight
                'w2': [0.0, 0.1, 0.2],  # MSE loss weight  
                'w3': [0.05, 0.1, 0.2, 0.3],  # Smoothness loss weight
                'w4': [0.005, 0.01, 0.02, 0.05]  # Bound penalty weight
            }
        }
        
        self.optimization_results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            # Fallback to JSON-like format
            with open(config_path, 'r') as f:
                content = f.read()
                # Simple YAML to dict conversion for basic cases
                config = {}
                for line in content.split('\n'):
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Parse values
                        if value == 'true':
                            value = True
                        elif value == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit():
                            value = float(value)
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        
                        config[key] = value
                return config
    
    def _save_config(self, config: Dict, save_path: str):
        """Save configuration to file."""
        try:
            import yaml
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def grid_search(self, 
                   param_grid: Dict[str, List[Any]], 
                   max_trials: Optional[int] = None,
                   evaluation_episodes: int = 5) -> List[Dict]:
        """Perform grid search over parameter combinations."""
        
        print(f"\n🔍 Starting Grid Search")
        print(f"📊 Parameter grid: {list(param_grid.keys())}")
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        print(f"🎯 Total combinations: {len(combinations)}")
        
        results = []
        for i, combination in enumerate(combinations):
            print(f"\n--- Trial {i+1}/{len(combinations)} ---")
            
            # Create config for this trial
            trial_config = self.base_config.copy()
            for key, value in zip(keys, combination):
                self._set_nested_config(trial_config, key, value)
            
            # Run training and evaluation
            result = self._run_trial(trial_config, trial_id=i+1, evaluation_episodes=evaluation_episodes)
            results.append(result)
            
            # Save intermediate results
            self._save_results(results, "grid_search_results.json")
        
        return results
    
    def random_search(self, 
                     param_distributions: Dict[str, List[Any]], 
                     n_iter: int = 20,
                     evaluation_episodes: int = 5) -> List[Dict]:
        """Perform random search over parameter distributions."""
        
        print(f"\n🎲 Starting Random Search")
        print(f"📊 Parameters: {list(param_distributions.keys())}")
        print(f"🎯 Iterations: {n_iter}")
        
        results = []
        for i in range(n_iter):
            print(f"\n--- Random Trial {i+1}/{n_iter} ---")
            
            # Sample random parameters
            trial_config = self.base_config.copy()
            for param_name, param_options in param_distributions.items():
                if param_name == 'loss_weights' and isinstance(param_options, dict):
                    # Handle nested loss weights
                    for weight_name, weight_options in param_options.items():
                        value = random.choice(weight_options)
                        self._set_nested_config(trial_config, f"loss_weights.{weight_name}", value)
                else:
                    value = random.choice(param_options)
                    self._set_nested_config(trial_config, param_name, value)
            
            # Run training and evaluation
            result = self._run_trial(trial_config, trial_id=i+1, evaluation_episodes=evaluation_episodes)
            results.append(result)
            
            # Save intermediate results
            self._save_results(results, "random_search_results.json")
        
        return results
    
    def _set_nested_config(self, config: Dict, key: str, value: Any):
        """Set nested configuration value using dot notation."""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _run_trial(self, config: Dict, trial_id: int, evaluation_episodes: int = 5) -> Dict:
        """Run a single training trial with given configuration."""
        
        # Create trial directory
        trial_dir = self.results_dir / f"trial_{trial_id:03d}"
        trial_dir.mkdir(exist_ok=True)
        
        # Save trial configuration
        config_path = trial_dir / "config.yaml"
        self._save_config(config, str(config_path))
        
        print(f"📁 Trial directory: {trial_dir}")
        
        # Extract key parameters for logging
        trial_params = {
            'learning_rate': config.get('learning_rate', 'unknown'),
            'batch_size': config.get('batch_size', 'unknown'),
            'epochs': config.get('epochs', 'unknown'),
            'frame_stack': config.get('frame_stack', 'unknown'),
            'section': config.get('section', 'unknown')
        }
        
        print(f"🔧 Parameters: {trial_params}")
        
        try:
            # Run training - construct command based on existing training infrastructure
            # Force correct data directory regardless of config
            train_cmd = [
                sys.executable, "-m", "passive_walker.bc.train",
                "--data", "data/fsm_demos",  # Always use correct data directory
                "--section", config['section'],
                "--epochs", str(config['epochs']),
                "--batch", str(config['batch_size']),
                "--lr", str(config['learning_rate']),
                "--frame-stack", str(config.get('frame_stack', 1)),
                "--backend", config.get('backend', 'torch'),
                "--seed", str(config.get('seed', 123)),
                "--save-dir", str(trial_dir)
            ]
            
            # Add loss weights if specified
            if 'loss_weights' in config:
                weights = config['loss_weights']
                train_cmd.extend([
                    "--w1", str(weights.get('w1', 1.0)),
                    "--w2", str(weights.get('w2', 0.0)),
                    "--w3", str(weights.get('w3', 0.1)),
                    "--w4", str(weights.get('w4', 0.01))
                ])
            
            print(f"🚀 Running training: {' '.join(train_cmd)}")
            
            # Run training
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                print(f"❌ Training failed: {result.stderr}")
                return {
                    'trial_id': trial_id,
                    'config': trial_params,
                    'success': False,
                    'error': result.stderr,
                    'training_time': None,
                    'metrics': None
                }
            
            # Find the trained model
            model_path, meta_path = self._find_trial_model(trial_dir)
            if not model_path:
                return {
                    'trial_id': trial_id,
                    'config': trial_params,
                    'success': False,
                    'error': "No model checkpoint found",
                    'training_time': None,
                    'metrics': None
                }
            
            # Run evaluation using our analysis tools
            eval_result = self._evaluate_model(model_path, meta_path, evaluation_episodes)
            
            result_data = {
                'trial_id': trial_id,
                'config': trial_params,
                'success': True,
                'model_path': str(model_path),
                'meta_path': str(meta_path),
                'training_time': self._extract_training_time(result.stdout),
                'metrics': eval_result,
                'trial_dir': str(trial_dir)
            }
            
            print(f"✅ Trial {trial_id} completed successfully")
            return result_data
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Trial {trial_id} timed out")
            return {
                'trial_id': trial_id,
                'config': trial_params,
                'success': False,
                'error': "Training timed out",
                'training_time': None,
                'metrics': None
            }
        except Exception as e:
            print(f"❌ Trial {trial_id} failed with error: {e}")
            return {
                'trial_id': trial_id,
                'config': trial_params,
                'success': False,
                'error': str(e),
                'training_time': None,
                'metrics': None
            }
    
    def _find_trial_model(self, trial_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find the trained model checkpoint in trial directory."""
        # Look for common checkpoint patterns
        patterns = [
            "*.pt",
            "checkpoints/*.pt", 
            "**/*checkpoint*.pt"
        ]
        
        for pattern in patterns:
            matches = list(trial_dir.glob(pattern))
            if matches:
                model_path = matches[0]
                meta_path = model_path.with_suffix('.json').with_name(
                    model_path.stem + '_meta.json'
                )
                if meta_path.exists():
                    return model_path, meta_path
        
        return None, None
    
    def _evaluate_model(self, model_path: str, meta_path: str, episodes: int = 5) -> Dict:
        """Evaluate trained model using analysis tools."""
        try:
            # Use our analysis runner to evaluate the model
            analysis_cmd = [
                sys.executable, "analysis_code/codes/run_analysis.py",
                "--ckpt", model_path,
                "--meta", meta_path,
                "--episodes", str(episodes),
                "--output-dir", str(Path(model_path).parent / "evaluation")
            ]
            
            result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Try to parse results from analysis output
                return {
                    'evaluation_success': True,
                    'raw_output': result.stdout,
                    'performance_score': self._parse_performance_score(result.stdout)
                }
            else:
                return {
                    'evaluation_success': False,
                    'error': result.stderr,
                    'performance_score': 0.0
                }
                
        except Exception as e:
            return {
                'evaluation_success': False,
                'error': str(e),
                'performance_score': 0.0
            }
    
    def _parse_performance_score(self, output: str) -> float:
        """Parse performance score from analysis output."""
        import re
        
        # Try to find "Overall Performance" improvement percentage
        overall_match = re.search(r'Overall Performance.*?([+-]?\d+\.?\d*)\s*%', output)
        if overall_match:
            return float(overall_match.group(1))
        
        # Try to find any improvement percentage
        improvement_match = re.search(r'improvement.*?([+-]?\d+\.?\d*)\s*%', output, re.IGNORECASE)
        if improvement_match:
            return float(improvement_match.group(1))
        
        # Try to extract from reward comparison
        reward_match = re.search(r'([+-]?\d+\.?\d*)\s*%\s*improvement', output, re.IGNORECASE)
        if reward_match:
            return float(reward_match.group(1))
        
        # Fallback: look for any percentage
        matches = re.findall(r'([+-]?\d+\.?\d*)%', output)
        if matches:
            return float(matches[0])
        
        return 0.0
    
    def _extract_training_time(self, output: str) -> Optional[float]:
        """Extract training time from output."""
        import re
        time_match = re.search(r'(\d+\.?\d*)\s*seconds', output)
        if time_match:
            return float(time_match.group(1))
        return None
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save optimization results to JSON file."""
        results_path = self.results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    def get_best_results(self, results: List[Dict], metric: str = 'performance_score') -> Dict:
        """Get the best result based on specified metric."""
        successful_results = [r for r in results if r.get('success', False) and r.get('metrics', {}).get('evaluation_success', False)]
        
        if not successful_results:
            print("⚠️ No successful results found")
            return {}
        
        best_result = max(successful_results, 
                         key=lambda x: x.get('metrics', {}).get(metric, 0.0))
        
        print(f"\n🏆 Best Result (Trial {best_result['trial_id']}):")
        print(f"   Config: {best_result['config']}")
        print(f"   {metric}: {best_result['metrics'].get(metric, 'N/A')}")
        
        return best_result
    
    def optimize(self, 
                 method: str = "random", 
                 search_space: Optional[Dict] = None,
                 max_trials: int = 20,
                 evaluation_episodes: int = 5) -> Dict:
        """Main optimization interface."""
        
        if search_space is None:
            search_space = self.search_spaces
        
        print(f"\n🚀 Starting Phase 3 Hyperparameter Optimization")
        print(f"📋 Method: {method}")
        print(f"📊 Search space: {list(search_space.keys())}")
        print(f"🎯 Max trials: {max_trials}")
        
        if method == "grid":
            results = self.grid_search(search_space, max_trials=max_trials, evaluation_episodes=evaluation_episodes)
        elif method == "random":
            results = self.random_search(search_space, n_iter=max_trials, evaluation_episodes=evaluation_episodes)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Find and return best result
        best_result = self.get_best_results(results)
        
        # Save final results
        self._save_results(results, f"{method}_search_final.json")
        
        return {
            'all_results': results,
            'best_result': best_result,
            'optimization_summary': {
                'total_trials': len(results),
                'successful_trials': len([r for r in results if r.get('success', False)]),
                'method': method,
                'search_space_keys': list(search_space.keys())
            }
        }
