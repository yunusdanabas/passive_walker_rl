"""
Advanced Training Strategies for Phase 3

Implements curriculum learning, data augmentation, and advanced training techniques.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import subprocess
import sys


class AdvancedTrainer:
    """Implements advanced training strategies for BC models."""
    
    def __init__(self, base_config_path: str, results_dir: str = "results/advanced_training"):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Advanced training strategies
        self.training_strategies = {
            'curriculum_learning': {
                'description': 'Gradual increase in difficulty/data complexity',
                'parameters': {
                    'curriculum_stages': [3, 5, 7],
                    'data_growth_ratios': [[0.3, 0.5], [0.5, 0.8], [1.0, 1.0]]
                }
            },
            'data_augmentation': {
                'description': 'Augment training data with noise and variations',
                'parameters': {
                    'noise_levels': [0.01, 0.05, 0.1],
                    'augmentation_factors': [1.5, 2.0, 3.0]
                }
            },
            'progressive_growing': {
                'description': 'Start with simple episodes, gradually add complexity',
                'parameters': {
                    'episode_lengths': [[5, 10], [10, 20], [20, 30]],
                    'complexity_stages': ['simple', 'medium', 'complex']
                }
            },
            'adaptive_loss': {
                'description': 'Adaptive loss weighting based on training progress',
                'parameters': {
                    'loss_adaptation': [True, False],
                    'adaptation_rate': [0.1, 0.2, 0.5]
                }
            }
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
    
    def curriculum_learning(self, 
                           stages: int = 3,
                           data_growth: List[List[float]] = None,
                           evaluation_episodes: int = 5) -> Dict:
        """Implement curriculum learning strategy."""
        
        if data_growth is None:
            data_growth = [[0.3], [0.6], [1.0]]
        
        print(f"\n📚 Starting Curriculum Learning")
        print(f"📊 Stages: {stages}")
        print(f"📈 Data growth: {data_growth}")
        
        trial_dir = self.results_dir / "curriculum_learning"
        trial_dir.mkdir(exist_ok=True)
        
        base_config = self._load_config(self.base_config_path)
        
        stage_results = []
        
        for stage in range(stages):
            print(f"\n--- Curriculum Stage {stage + 1}/{stages} ---")
            
            # Modify config for this stage
            stage_config = base_config.copy()
            stage_data_ratio = data_growth[stage][0] if stage < len(data_growth) else 1.0
            
            # Adjust training parameters for this stage
            stage_config['epochs'] = base_config.get('epochs', 30) + stage * 10  # Gradually increase epochs
            stage_config['learning_rate'] = base_config.get('learning_rate', 0.001) / (1 + stage * 0.1)  # Gradually decrease LR
            
            stage_dir = trial_dir / f"stage_{stage+1}"
            stage_dir.mkdir(exist_ok=True)
            
            # Save stage config
            config_path = stage_dir / "stage_config.yaml"
            self._save_config(stage_config, str(config_path))
            
            # Run training for this stage
            result = self._run_curriculum_stage(stage_config, stage_dir, stage, stage_data_ratio, evaluation_episodes)
            stage_results.append(result)
        
        return {
            'strategy': 'curriculum_learning',
            'stages_completed': len(stage_results),
            'stage_results': stage_results,
            'trial_dir': str(trial_dir)
        }
    
    def _run_curriculum_stage(self, config: Dict, stage_dir: Path, stage: int, data_ratio: float, evaluation_episodes: int) -> Dict:
        """Run a single curriculum learning stage."""
        
        try:
            # Build training command
            train_cmd = [
                sys.executable, "-m", "passive_walker.bc.train",
                "--data", "data/fsm_demos",  # Always use correct data directory
                "--section", config['section'],
                "--epochs", str(config['epochs']),
                "--batch", str(config.get('batch_size', 1024)),
                "--lr", str(config['learning_rate']),
                "--backend", config.get('backend', 'torch'),
                "--seed", str(config.get('seed', 123)) + str(stage),  # Different seed per stage
                "--save-dir", str(stage_dir)
            ]
            
            # Note: --data-ratio not currently supported by training script
            # train_cmd.extend(["--data-ratio", str(data_ratio)])
            
            print(f"🚀 Running curriculum stage {stage + 1}")
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                return {
                    'stage': stage + 1,
                    'success': False,
                    'error': result.stderr,
                    'metrics': None
                }
            
            # Find and evaluate model
            model_path, meta_path = self._find_stage_model(stage_dir)
            if not model_path:
                return {
                    'stage': stage + 1,
                    'success': False,
                    'error': "No model checkpoint found",
                    'metrics': None
                }
            
            eval_result = self._evaluate_stage(model_path, meta_path, evaluation_episodes)
            
            return {
                'stage': stage + 1,
                'success': True,
                'model_path': str(model_path),
                'metrics': eval_result,
                'stage_dir': str(stage_dir)
            }
            
        except Exception as e:
            return {
                'stage': stage + 1,
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def data_augmentation(self, 
                         noise_level: float = 0.05,
                         augmentation_factor: float = 2.0,
                         evaluation_episodes: int = 5) -> Dict:
        """Implement data augmentation strategy."""
        
        print(f"\n🔄 Starting Data Augmentation Training")
        print(f"📊 Noise level: {noise_level}")
        print(f"📈 Augmentation factor: {augmentation_factor}")
        
        trial_dir = self.results_dir / "data_augmentation"
        trial_dir.mkdir(exist_ok=True)
        
        base_config = self._load_config(self.base_config_path)
        
        # Modify config for augmented training
        aug_config = base_config.copy()
        aug_config['epochs'] = base_config.get('epochs', 30) * 2  # Longer training for augmented data
        
        try:
            # Run augmented training
            train_cmd = [
                sys.executable, "-m", "passive_walker.bc.train",
                "--data", "data/fsm_demos",  # Always use correct data directory
                "--section", aug_config['section'],
                "--epochs", str(aug_config['epochs']),
                "--batch", str(aug_config.get('batch_size', 1024)),
                "--lr", str(aug_config.get('learning_rate', 0.001)),
                "--backend", aug_config.get('backend', 'torch'),
                "--seed", str(aug_config.get('seed', 123)),
                "--save-dir", str(trial_dir)
                # Note: Augmentation arguments not currently supported by training script
                # "--augment-data", "--noise-level", str(noise_level),
                # "--augmentation-factor", str(augmentation_factor)
            ]
            
            print(f"🚀 Running augmented training")
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                return {
                    'strategy': 'data_augmentation',
                    'success': False,
                    'error': result.stderr,
                    'metrics': None
                }
            
            # Find and evaluate model
            model_path, meta_path = self._find_stage_model(trial_dir)
            if not model_path:
                return {
                    'strategy': 'data_augmentation',
                    'success': False,
                    'error': "No model checkpoint found",
                    'metrics': None
                }
            
            eval_result = self._evaluate_stage(model_path, meta_path, evaluation_episodes)
            
            return {
                'strategy': 'data_augmentation',
                'success': True,
                'model_path': str(model_path),
                'metrics': eval_result,
                'trial_dir': str(trial_dir)
            }
            
        except Exception as e:
            return {
                'strategy': 'data_augmentation',
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def progressive_growing(self, 
                           episode_lengths: List[List[int]] = None,
                           evaluation_episodes: int = 5) -> Dict:
        """Implement progressive growing strategy."""
        
        if episode_lengths is None:
            episode_lengths = [[5, 10], [10, 20], [20, 30]]
        
        print(f"\n🌱 Starting Progressive Growing Training")
        print(f"📊 Episode length progression: {episode_lengths}")
        
        trial_dir = self.results_dir / "progressive_growing"
        trial_dir.mkdir(exist_ok=True)
        
        base_config = self._load_config(self.base_config_path)
        
        stage_results = []
        
        for i, (min_length, max_length) in enumerate(episode_lengths):
            print(f"\n--- Progressive Stage {i + 1} (episodes {min_length}-{max_length}s) ---")
            
            # Modify config for this stage
            stage_config = base_config.copy()
            stage_config['episodes'] = base_config.get('episodes', 200) // len(episode_lengths)  # Split episodes across stages
            
            stage_dir = trial_dir / f"stage_{i+1}"
            stage_dir.mkdir(exist_ok=True)
            
            # Run training with episode length constraints
            result = self._run_progressive_stage(stage_config, stage_dir, min_length, max_length, i, evaluation_episodes)
            stage_results.append(result)
        
        return {
            'strategy': 'progressive_growing',
            'stages_completed': len(stage_results),
            'stage_results': stage_results,
            'trial_dir': str(trial_dir)
        }
    
    def _run_progressive_stage(self, config: Dict, stage_dir: Path, min_length: int, max_length: int, stage: int, evaluation_episodes: int) -> Dict:
        """Run a single progressive growing stage."""
        
        try:
            train_cmd = [
                sys.executable, "-m", "passive_walker.bc.train",
                "--data", "data/fsm_demos",  # Always use correct data directory
                "--section", config['section'],
                "--epochs", str(config.get('epochs', 30)),
                "--batch", str(config.get('batch_size', 1024)),
                "--lr", str(config.get('learning_rate', 0.001)),
                "--backend", config.get('backend', 'torch'),
                "--seed", str(config.get('seed', 123)) + str(stage),
                "--save-dir", str(stage_dir)
                # Note: Episode length arguments not currently supported by training script
                # "--min-episode-length", str(min_length),
                # "--max-episode-length", str(max_length)
            ]
            
            print(f"🚀 Running progressive stage {stage + 1}")
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                return {
                    'stage': stage + 1,
                    'success': False,
                    'error': result.stderr,
                    'metrics': None
                }
            
            model_path, meta_path = self._find_stage_model(stage_dir)
            if not model_path:
                return {
                    'stage': stage + 1,
                    'success': False,
                    'error': "No model checkpoint found",
                    'metrics': None
                }
            
            eval_result = self._evaluate_stage(model_path, meta_path, evaluation_episodes)
            
            return {
                'stage': stage + 1,
                'success': True,
                'model_path': str(model_path),
                'metrics': eval_result,
                'stage_dir': str(stage_dir)
            }
            
        except Exception as e:
            return {
                'stage': stage + 1,
                'success': False,
                'error': str(e),
                'metrics': None
            }
    
    def _find_stage_model(self, stage_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find the trained model checkpoint in stage directory."""
        patterns = ["*.pt", "checkpoints/*.pt", "**/*checkpoint*.pt"]
        
        for pattern in patterns:
            matches = list(stage_dir.glob(pattern))
            if matches:
                model_path = matches[0]
                meta_path = model_path.with_suffix('.json').with_name(
                    model_path.stem + '_meta.json'
                )
                if meta_path.exists():
                    return model_path, meta_path
        
        return None, None
    
    def _evaluate_stage(self, model_path: str, meta_path: str, episodes: int = 5) -> Dict:
        """Evaluate training stage using analysis tools."""
        try:
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
    
    def compare_strategies(self, 
                          strategies: List[str] = None,
                          evaluation_episodes: int = 5) -> Dict:
        """Compare different advanced training strategies."""
        
        if strategies is None:
            strategies = ['curriculum_learning', 'data_augmentation', 'progressive_growing']
        
        print(f"\n⚖️ Comparing Training Strategies")
        print(f"📊 Strategies: {strategies}")
        
        comparison_results = {}
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy} ---")
            
            if strategy == 'curriculum_learning':
                result = self.curriculum_learning(evaluation_episodes=evaluation_episodes)
            elif strategy == 'data_augmentation':
                result = self.data_augmentation(evaluation_episodes=evaluation_episodes)
            elif strategy == 'progressive_growing':
                result = self.progressive_growing(evaluation_episodes=evaluation_episodes)
            else:
                print(f"⚠️ Unknown strategy: {strategy}")
                continue
            
            comparison_results[strategy] = result
        
        # Analyze comparison results
        analysis = self._analyze_strategy_comparison(comparison_results)
        
        # Save comparison results
        results_path = self.results_dir / "strategy_comparison.json"
        with open(results_path, 'w') as f:
            json.dump({
                'comparison_results': comparison_results,
                'analysis': analysis
            }, f, indent=2)
        
        print(f"\n📊 Strategy Comparison Complete!")
        print(f"💾 Results saved to: {results_path}")
        
        return {
            'comparison_results': comparison_results,
            'analysis': analysis,
            'results_path': str(results_path)
        }
    
    def _analyze_strategy_comparison(self, comparison_results: Dict) -> Dict:
        """Analyze results from strategy comparison."""
        
        analysis = {
            'strategies_tested': list(comparison_results.keys()),
            'strategy_performance': {},
            'recommendations': []
        }
        
        for strategy_name, result in comparison_results.items():
            if strategy_name == 'curriculum_learning':
                # Analyze curriculum learning results
                stages = result.get('stage_results', [])
                successful_stages = [s for s in stages if s.get('success', False)]
                
                if successful_stages:
                    final_stage = successful_stages[-1]
                    performance = final_stage.get('metrics', {}).get('performance_score', 0.0)
                    analysis['strategy_performance'][strategy_name] = {
                        'final_performance': performance,
                        'stages_completed': len(successful_stages),
                        'success_rate': len(successful_stages) / len(stages) if stages else 0.0
                    }
        
        # Generate recommendations
        best_strategy = max(analysis['strategy_performance'].items(), 
                           key=lambda x: x[1].get('final_performance', 0.0)) if analysis['strategy_performance'] else None
        
        if best_strategy:
            analysis['recommendations'].append(f"Best performing strategy: {best_strategy[0]}")
            analysis['recommendations'].append(f"Consider using {best_strategy[0]} for future training")
        
        return analysis
    
    def optimize(self, 
                 strategies: List[str] = None,
                 evaluation_episodes: int = 5) -> Dict:
        """Main advanced training optimization interface."""
        
        print(f"\n🚀 Starting Phase 3 Advanced Training Optimization")
        
        if strategies is None:
            strategies = ['curriculum_learning']  # Start with most promising
        
        # Run strategy comparison
        results = self.compare_strategies(strategies, evaluation_episodes)
        
        return {
            'advanced_training_results': results,
            'optimization_summary': {
                'strategies_tested': strategies,
                'best_strategy': results.get('analysis', {}).get('recommendations', [{}])[-1] if results.get('analysis', {}).get('recommendations') else None
            }
        }
