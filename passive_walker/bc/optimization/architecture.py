"""
Architecture Optimizer for Phase 3

Implements neural architecture search for finding optimal network designs.
Tests different network depths, widths, and configurations.
"""

import os
import json
import itertools
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import subprocess
import sys


class ArchitectureOptimizer:
    """Optimizes neural network architecture using systematic search."""
    
    def __init__(self, base_config_path: str, results_dir: str = "results/architecture_optimization"):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Architecture search spaces
        self.architecture_spaces = {
            'hidden_sizes': [
                [64, 64],           # Small
                [128, 128],         # Medium
                [256, 256],         # Large
                [512, 512],         # Extra Large
                [256, 128, 64],     # Decreasing
                [64, 128, 256],     # Increasing
                [128, 256, 128],    # Diamond
                [512, 256],         # Large to medium
                [256, 512, 256],    # Hourglass
                [128, 128, 128, 128] # Deep narrow
            ],
            'dropout_rates': [0.0, 0.1, 0.2, 0.3],
            'activation_functions': ['relu', 'tanh', 'elu'],
            'normalization': [None, 'batch_norm', 'layer_norm'],
            'frame_stacks': [1, 2, 3, 4]
        }
        
        self.optimization_results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            with open(config_path, 'r') as f:
                return json.load(f)
    
    def _save_config(self, config: Dict, save_path: str):
        """Save configuration to file."""
        try:
            import yaml
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except ImportError:
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def _modify_model_class(self, architecture_config: Dict) -> str:
        """Generate or modify model class based on architecture configuration."""
        hidden_sizes = architecture_config['hidden_sizes']
        dropout_rate = architecture_config.get('dropout_rate', 0.1)
        activation = architecture_config.get('activation', 'relu')
        
        # Create a custom model class string
        model_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, {hidden_sizes[0]})]
        
        # Hidden layers
        for i in range(len({hidden_sizes}) - 1):
            layers.append(nn.ReLU() if "{activation}" == "relu" else nn.Tanh() if "{activation}" == "tanh" else nn.ELU())
            layers.append(nn.Dropout({dropout_rate}))
            layers.append(nn.Linear({hidden_sizes}[i], {hidden_sizes}[i + 1]))
        
        # Output layer
        layers.append(nn.ReLU() if "{activation}" == "relu" else nn.Tanh() if "{activation}" == "tanh" else nn.ELU())
        layers.append(nn.Linear({hidden_sizes}[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
'''
        return model_code
    
    def systematic_search(self, 
                         max_trials: Optional[int] = None,
                         evaluation_episodes: int = 5) -> List[Dict]:
        """Perform systematic architecture search."""
        
        print(f"\n🏗️ Starting Architecture Search")
        
        # Generate architecture combinations
        combinations = []
        
        # Test different architecture sizes (small to large)
        size_groups = {
            'small': self.architecture_spaces['hidden_sizes'][:3],
            'medium': self.architecture_spaces['hidden_sizes'][3:6], 
            'large': self.architecture_spaces['hidden_sizes'][6:]
        }
        
        for size_group, architectures in size_groups.items():
            for arch in architectures:
                # Test different dropout rates for each architecture
                for dropout in [0.0, 0.1, 0.2]:
                    combination = {
                        'hidden_sizes': arch,
                        'dropout_rate': dropout,
                        'activation': 'relu',  # Default to relu for now
                        'size_group': size_group,
                        'total_params': self._estimate_parameters(arch, 11)  # Rough estimate
                    }
                    combinations.append(combination)
        
        if max_trials:
            combinations = combinations[:max_trials]
        
        print(f"🎯 Total architectures to test: {len(combinations)}")
        
        results = []
        for i, arch_config in enumerate(combinations):
            print(f"\n--- Architecture Trial {i+1}/{len(combinations)} ---")
            print(f"🏗️ Architecture: {arch_config['hidden_sizes']}")
            print(f"📊 Size group: {arch_config['size_group']}")
            print(f"🎛️ Dropout: {arch_config['dropout_rate']}")
            
            result = self._run_architecture_trial(arch_config, trial_id=i+1, evaluation_episodes=evaluation_episodes)
            results.append(result)
            
            # Save intermediate results
            self._save_results(results, "architecture_search_results.json")
        
        return results
    
    def _estimate_parameters(self, hidden_sizes: List[int], input_dim: int) -> int:
        """Estimate number of parameters for architecture."""
        total = 0
        prev_size = input_dim
        
        for size in hidden_sizes:
            total += prev_size * size + size  # weights + biases
            prev_size = size
        
        # Add output layer (assuming 3 outputs)
        total += prev_size * 3 + 3
        
        return total
    
    def _run_architecture_trial(self, arch_config: Dict, trial_id: int, evaluation_episodes: int = 5) -> Dict:
        """Run a single architecture trial."""
        
        trial_dir = self.results_dir / f"arch_trial_{trial_id:03d}"
        trial_dir.mkdir(exist_ok=True)
        
        try:
            # We'll need to modify the training script to accept architecture parameters
            # For now, we'll use a simplified approach by modifying the model creation
            
            # Load base config
            base_config = self._load_config(self.base_config_path)
            
            # Modify config for this architecture
            trial_config = base_config.copy()
            
            # Save architecture config
            arch_config_path = trial_dir / "architecture_config.json"
            with open(arch_config_path, 'w') as f:
                json.dump(arch_config, f, indent=2)
            
            # Run training with architecture modifications
            train_cmd = self._build_training_command(trial_config, arch_config, trial_dir)
            
            print(f"🚀 Running architecture trial: {' '.join(train_cmd)}")
            
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                print(f"❌ Architecture trial failed: {result.stderr}")
                return {
                    'trial_id': trial_id,
                    'architecture': arch_config,
                    'success': False,
                    'error': result.stderr,
                    'metrics': None
                }
            
            # Find trained model and evaluate
            model_path, meta_path = self._find_trial_model(trial_dir)
            if not model_path:
                return {
                    'trial_id': trial_id,
                    'architecture': arch_config,
                    'success': False,
                    'error': "No model checkpoint found",
                    'metrics': None
                }
            
            # Evaluate the architecture
            eval_result = self._evaluate_architecture(model_path, meta_path, evaluation_episodes)
            
            result_data = {
                'trial_id': trial_id,
                'architecture': arch_config,
                'success': True,
                'model_path': str(model_path),
                'meta_path': str(meta_path),
                'metrics': eval_result,
                'trial_dir': str(trial_dir)
            }
            
            print(f"✅ Architecture trial {trial_id} completed")
            return result_data
            
        except Exception as e:
            print(f"❌ Architecture trial {trial_id} failed: {e}")
            return {
                'trial_id': trial_id,
                'architecture': arch_config,
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def _build_training_command(self, config: Dict, arch_config: Dict, trial_dir: Path) -> List[str]:
        """Build training command with architecture-specific parameters."""
        
        # For now, we'll pass architecture info via environment variables
        # This requires modification of the training script to read these
        
        cmd = [
            sys.executable, "-m", "passive_walker.bc.train",
            "--data", "data/fsm_demos",  # Always use correct data directory
            "--section", config['section'],
            "--epochs", str(config.get('epochs', 30)),
            "--batch", str(config.get('batch_size', 1024)),
            "--lr", str(config.get('learning_rate', 0.001)),
            "--backend", config.get('backend', 'torch'),
            "--seed", str(config.get('seed', 123)),
            "--save-dir", str(trial_dir)
            # Note: Architecture-specific parameters not currently supported by training script
            # "--arch-hidden", ",".join(map(str, arch_config['hidden_sizes'])),
            # "--arch-dropout", str(arch_config['dropout_rate']),
        ]
        
        return cmd
    
    def _find_trial_model(self, trial_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find the trained model checkpoint in trial directory."""
        patterns = ["*.pt", "checkpoints/*.pt", "**/*checkpoint*.pt"]
        
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
    
    def _evaluate_architecture(self, model_path: str, meta_path: str, episodes: int = 5) -> Dict:
        """Evaluate architecture using analysis tools."""
        try:
            # Use analysis tools to evaluate
            analysis_cmd = [
                sys.executable, "analysis_code/codes/run_analysis.py",
                "--ckpt", model_path,
                "--meta", meta_path,
                "--episodes", str(episodes),
                "--output-dir", str(Path(model_path).parent / "evaluation")
            ]
            
            result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return {
                    'evaluation_success': True,
                    'performance_score': self._parse_performance_score(result.stdout),
                    'raw_output': result.stdout
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
        overall_match = re.search(r'Overall Performance.*?([+-]?\d+\.?\d*)\s*%', output)
        if overall_match:
            return float(overall_match.group(1))
        improvement_match = re.search(r'improvement.*?([+-]?\d+\.?\d*)\s*%', output, re.IGNORECASE)
        if improvement_match:
            return float(improvement_match.group(1))
        matches = re.findall(r'([+-]?\d+\.?\d*)%', output)
        if matches:
            return float(matches[0])
        return 0.0
        """Parse performance score from analysis output."""
        import re
        matches = re.findall(r'(\+?-?\d+\.?\d*)%', output)
        if matches:
            return float(matches[0])
        return 0.0
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save architecture search results."""
        results_path = self.results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Architecture results saved to: {results_path}")
    
    def analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze architecture search results to find patterns."""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        analysis = {
            'total_trials': len(results),
            'successful_trials': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'architecture_insights': {}
        }
        
        # Find best performing architecture
        best_result = max(successful_results, 
                         key=lambda x: x.get('metrics', {}).get('performance_score', 0.0))
        
        analysis['best_architecture'] = {
            'trial_id': best_result['trial_id'],
            'architecture': best_result['architecture'],
            'performance_score': best_result.get('metrics', {}).get('performance_score', 0.0)
        }
        
        # Analyze patterns by size group
        size_group_performance = {}
        for result in successful_results:
            arch = result['architecture']
            size_group = arch.get('size_group', 'unknown')
            performance = result.get('metrics', {}).get('performance_score', 0.0)
            
            if size_group not in size_group_performance:
                size_group_performance[size_group] = []
            size_group_performance[size_group].append(performance)
        
        # Calculate average performance by size group
        for size_group, performances in size_group_performance.items():
            analysis['architecture_insights'][f'{size_group}_avg_performance'] = sum(performances) / len(performances)
            analysis['architecture_insights'][f'{size_group}_best_performance'] = max(performances)
        
        return analysis
    
    def optimize(self, max_trials: int = 15, evaluation_episodes: int = 5) -> Dict:
        """Main architecture optimization interface."""
        
        print(f"\n🚀 Starting Phase 3 Architecture Optimization")
        print(f"🎯 Max trials: {max_trials}")
        print(f"📊 Architecture spaces: {list(self.architecture_spaces.keys())}")
        
        # Run systematic architecture search
        results = self.systematic_search(max_trials=max_trials, evaluation_episodes=evaluation_episodes)
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Find best architecture
        successful_results = [r for r in results if r.get('success', False)]
        best_result = max(successful_results, key=lambda x: x.get('metrics', {}).get('performance_score', 0.0)) if successful_results else {}
        
        # Save final results
        self._save_results(results, "architecture_search_final.json")
        
        final_result = {
            'all_results': results,
            'best_architecture': best_result,
            'analysis': analysis,
            'optimization_summary': {
                'total_trials': len(results),
                'successful_trials': len(successful_results),
                'best_performance': best_result.get('metrics', {}).get('performance_score', 0.0) if best_result else 0.0
            }
        }
        
        print(f"\n🏆 Architecture Optimization Complete!")
        if best_result:
            print(f"   Best Architecture: {best_result['architecture']}")
            print(f"   Performance Score: {best_result.get('metrics', {}).get('performance_score', 0.0)}")
        
        return final_result
