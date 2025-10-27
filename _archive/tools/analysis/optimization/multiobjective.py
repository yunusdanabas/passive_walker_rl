"""
Multi-Objective Optimizer for Phase 3

Balances performance, efficiency, and robustness objectives.
Uses Pareto optimization to find trade-offs between competing objectives.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import subprocess
import sys


class MultiObjectiveOptimizer:
    """Optimizes models across multiple objectives using Pareto frontier analysis."""
    
    def __init__(self, base_config_path: str, results_dir: str = "results/multiobjective_optimization"):
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define optimization objectives
        self.objectives = {
            'performance': {
                'description': 'Model performance (reward/forward progress)',
                'weight': 0.4,
                'minimize': False  # Maximize performance
            },
            'efficiency': {
                'description': 'Training efficiency (speed/sample efficiency)',
                'weight': 0.3,
                'minimize': True  # Minimize training time/data requirements
            },
            'robustness': {
                'description': 'Model robustness across scenarios',
                'weight': 0.3,
                'minimize': False  # Maximize robustness
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
    
    def evaluate_objectives(self, 
                          model_path: str, 
                          meta_path: str, 
                          training_time: Optional[float] = None,
                          evaluation_episodes: int = 5) -> Dict[str, float]:
        """Evaluate model across multiple objectives."""
        
        objectives = {}
        
        try:
            # 1. Performance objective - use analysis tools
            performance_result = self._evaluate_performance(model_path, meta_path, evaluation_episodes)
            objectives['performance'] = performance_result.get('performance_score', 0.0)
            
            # 2. Efficiency objective - based on training metrics
            efficiency_result = self._evaluate_efficiency(model_path, meta_path, training_time)
            objectives['efficiency'] = efficiency_result.get('efficiency_score', 0.0)
            
            # 3. Robustness objective - use robustness testing
            robustness_result = self._evaluate_robustness(model_path, meta_path, evaluation_episodes)
            objectives['robustness'] = robustness_result.get('robustness_score', 0.0)
            
        except Exception as e:
            print(f"⚠️ Error evaluating objectives: {e}")
            objectives = {
                'performance': 0.0,
                'efficiency': 0.0,
                'robustness': 0.0
            }
        
        return objectives
    
    def _evaluate_performance(self, model_path: str, meta_path: str, episodes: int = 5) -> Dict:
        """Evaluate performance objective."""
        try:
            # Use basic performance analysis
            analysis_cmd = [
                sys.executable, "analysis_code/codes/run_analysis.py",
                "--ckpt", model_path,
                "--meta", meta_path,
                "--episodes", str(episodes),
                "--output-dir", str(Path(model_path).parent / "performance_eval")
            ]
            
            result = subprocess.run(analysis_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                performance_score = self._parse_performance_score(result.stdout)
                return {
                    'evaluation_success': True,
                    'performance_score': performance_score,
                    'raw_output': result.stdout
                }
            else:
                return {
                    'evaluation_success': False,
                    'performance_score': 0.0,
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'evaluation_success': False,
                'performance_score': 0.0,
                'error': str(e)
            }
    
    def _evaluate_efficiency(self, model_path: str, meta_path: str, training_time: Optional[float]) -> Dict:
        """Evaluate efficiency objective."""
        try:
            # Load model metadata to get training info
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Calculate efficiency based on training metrics
            epochs = meta.get('epochs', 30)
            input_dim = meta.get('input_dim', 11)
            output_dim = meta.get('output_dim', 3)
            
            # Estimate model parameters (rough)
            estimated_params = input_dim * 512 * 3 + output_dim * 512  # Rough estimate
            param_efficiency = 1.0 / max(1, estimated_params / 1000)  # Normalize
            
            # Time efficiency (if available)
            if training_time:
                time_efficiency = 1.0 / max(1, training_time / 3600)  # Normalize by hour
            else:
                time_efficiency = 1.0 / max(1, epochs / 50)  # Use epoch count as proxy
            
            # Combined efficiency score
            efficiency_score = (param_efficiency * 0.6 + time_efficiency * 0.4)
            
            return {
                'efficiency_score': efficiency_score,
                'estimated_params': estimated_params,
                'training_time': training_time,
                'epochs': epochs
            }
            
        except Exception as e:
            return {
                'efficiency_score': 0.0,
                'error': str(e)
            }
    
    def _evaluate_robustness(self, model_path: str, meta_path: str, episodes: int = 3) -> Dict:
        """Evaluate robustness objective."""
        try:
            # Use extended physics variations for robustness testing
            robustness_cmd = [
                sys.executable, "analysis_code/codes/extended_physics_variations.py",
                "--ckpt", model_path,
                "--meta", meta_path,
                "--episodes", str(episodes),
                "--output-dir", str(Path(model_path).parent / "robustness_eval")
            ]
            
            result = subprocess.run(robustness_cmd, capture_output=True, text=True, timeout=900)
            
            if result.returncode == 0:
                # Parse robustness score from output
                robustness_score = self._parse_robustness_score(result.stdout)
                return {
                    'evaluation_success': True,
                    'robustness_score': robustness_score,
                    'raw_output': result.stdout
                }
            else:
                return {
                    'evaluation_success': False,
                    'robustness_score': 0.0,
                    'error': result.stderr
                }
                
        except Exception as e:
            return {
                'evaluation_success': False,
                'robustness_score': 0.0,
                'error': str(e)
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
        """Parse performance score from output."""
        import re
        # Look for improvement percentage or performance metrics
        matches = re.findall(r'(\+?-?\d+\.?\d*)%', output)
        if matches:
            return float(matches[0])
        return 0.0
    
    def _parse_robustness_score(self, output: str) -> float:
        """Parse robustness score from output."""
        import re
        # Look for success rates or robustness metrics
        success_matches = re.findall(r'success.*?(\d+\.?\d*)', output, re.IGNORECASE)
        if success_matches:
            # Average success rate across scenarios
            success_rates = [float(m) for m in success_matches]
            return sum(success_rates) / len(success_rates)
        return 0.0
    
    def calculate_pareto_score(self, objectives: Dict[str, float]) -> float:
        """Calculate weighted Pareto score from multiple objectives."""
        
        weighted_score = 0.0
        
        for obj_name, obj_config in self.objectives.items():
            if obj_name in objectives:
                value = objectives[obj_name]
                
                if obj_config['minimize']:
                    # Invert for minimization objectives
                    value = 1.0 / max(0.001, value)  # Avoid division by zero
                
                # Apply weight and accumulate
                weighted_score += obj_config['weight'] * value
        
        return weighted_score
    
    def find_pareto_frontier(self, results: List[Dict]) -> List[Dict]:
        """Find Pareto optimal solutions."""
        
        if not results:
            return []
        
        # Filter successful results with valid objectives
        valid_results = []
        for result in results:
            if (result.get('success', False) and 
                'objectives' in result and 
                all(obj in result['objectives'] for obj in self.objectives.keys())):
                valid_results.append(result)
        
        if not valid_results:
            return []
        
        pareto_frontier = []
        
        for candidate in valid_results:
            is_pareto_optimal = True
            candidate_obj = candidate['objectives']
            
            # Check if any other solution dominates this candidate
            for other in valid_results:
                if other == candidate:
                    continue
                
                other_obj = other['objectives']
                
                # Check if other dominates candidate
                if self._dominates(other_obj, candidate_obj, self.objectives):
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_frontier.append(candidate)
        
        # Sort by weighted Pareto score
        pareto_frontier.sort(key=lambda x: self.calculate_pareto_score(x['objectives']), reverse=True)
        
        return pareto_frontier
    
    def _dominates(self, obj_a: Dict[str, float], obj_b: Dict[str, float], objectives: Dict) -> bool:
        """Check if solution A dominates solution B."""
        
        better_in_at_least_one = False
        
        for obj_name, obj_config in objectives.items():
            if obj_name not in obj_a or obj_name not in obj_b:
                return False
            
            value_a = obj_a[obj_name]
            value_b = obj_b[obj_name]
            
            if obj_config['minimize']:
                # For minimization: A is better if A < B
                if value_a > value_b:
                    return False  # A is worse
                elif value_a < value_b:
                    better_in_at_least_one = True
            else:
                # For maximization: A is better if A > B
                if value_a < value_b:
                    return False  # A is worse
                elif value_a > value_b:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def optimize_multiobjective(self, 
                               candidate_configs: List[Dict],
                               evaluation_episodes: int = 5) -> Dict:
        """Main multi-objective optimization interface."""
        
        print(f"\n🎯 Starting Multi-Objective Optimization")
        print(f"📊 Objectives: {list(self.objectives.keys())}")
        print(f"🔍 Candidates: {len(candidate_configs)}")
        
        results = []
        
        for i, config in enumerate(candidate_configs):
            print(f"\n--- Evaluating Candidate {i+1}/{len(candidate_configs)} ---")
            
            # Run training with this configuration
            result = self._run_candidate_training(config, i+1, evaluation_episodes)
            results.append(result)
            
            # Save intermediate results
            self._save_results(results, "multiobjective_results.json")
        
        # Find Pareto frontier
        pareto_frontier = self.find_pareto_frontier(results)
        
        # Analyze results
        analysis = self._analyze_multiobjective_results(results, pareto_frontier)
        
        final_result = {
            'all_results': results,
            'pareto_frontier': pareto_frontier,
            'analysis': analysis,
            'optimization_summary': {
                'total_candidates': len(candidate_configs),
                'successful_candidates': len([r for r in results if r.get('success', False)]),
                'pareto_optimal_solutions': len(pareto_frontier),
                'objectives': list(self.objectives.keys())
            }
        }
        
        # Save final results
        self._save_results(results, "multiobjective_final.json")
        
        print(f"\n🏆 Multi-Objective Optimization Complete!")
        print(f"📊 Found {len(pareto_frontier)} Pareto optimal solutions")
        
        if pareto_frontier:
            best_solution = pareto_frontier[0]
            print(f"🏅 Best solution objectives: {best_solution['objectives']}")
        
        return final_result
    
    def _run_candidate_training(self, config: Dict, candidate_id: int, evaluation_episodes: int) -> Dict:
        """Run training for a candidate configuration."""
        
        candidate_dir = self.results_dir / f"candidate_{candidate_id:03d}"
        candidate_dir.mkdir(exist_ok=True)
        
        base_config = self._load_config(self.base_config_path)
        
        # Merge candidate config with base config
        trial_config = base_config.copy()
        trial_config.update(config)
        
        try:
            # Run training
            train_cmd = [
                sys.executable, "-m", "passive_walker.bc.train",
                "--data", "data/fsm_demos",  # Always use correct data directory
                "--section", trial_config['section'],
                "--epochs", str(trial_config.get('epochs', 30)),
                "--batch", str(trial_config.get('batch_size', 1024)),
                "--lr", str(trial_config.get('learning_rate', 0.001)),
                "--backend", trial_config.get('backend', 'torch'),
                "--seed", str(trial_config.get('seed', 123)) + str(candidate_id),
                "--save-dir", str(candidate_dir)
            ]
            
            print(f"🚀 Training candidate {candidate_id}")
            start_time = self._get_time()
            
            result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=3600)
            
            training_time = self._get_time() - start_time if self._get_time() else None
            
            if result.returncode != 0:
                return {
                    'candidate_id': candidate_id,
                    'config': config,
                    'success': False,
                    'error': result.stderr,
                    'objectives': None
                }
            
            # Find trained model
            model_path, meta_path = self._find_candidate_model(candidate_dir)
            if not model_path:
                return {
                    'candidate_id': candidate_id,
                    'config': config,
                    'success': False,
                    'error': "No model checkpoint found",
                    'objectives': None
                }
            
            # Evaluate across all objectives
            objectives = self.evaluate_objectives(str(model_path), str(meta_path), training_time)
            pareto_score = self.calculate_pareto_score(objectives)
            
            return {
                'candidate_id': candidate_id,
                'config': config,
                'success': True,
                'model_path': str(model_path),
                'meta_path': str(meta_path),
                'training_time': training_time,
                'objectives': objectives,
                'pareto_score': pareto_score,
                'candidate_dir': str(candidate_dir)
            }
            
        except Exception as e:
            return {
                'candidate_id': candidate_id,
                'config': config,
                'success': False,
                'error': str(e),
                'objectives': None
            }
    
    def _find_candidate_model(self, candidate_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """Find the trained model checkpoint."""
        patterns = ["*.pt", "checkpoints/*.pt", "**/*checkpoint*.pt"]
        
        for pattern in patterns:
            matches = list(candidate_dir.glob(pattern))
            if matches:
                model_path = matches[0]
                meta_path = model_path.with_suffix('.json').with_name(
                    model_path.stem + '_meta.json'
                )
                if meta_path.exists():
                    return model_path, meta_path
        
        return None, None
    
    def _get_time(self) -> Optional[float]:
        """Get current time for timing measurements."""
        try:
            import time
            return time.time()
        except:
            return None
    
    def _analyze_multiobjective_results(self, results: List[Dict], pareto_frontier: List[Dict]) -> Dict:
        """Analyze multi-objective optimization results."""
        
        successful_results = [r for r in results if r.get('success', False)]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        analysis = {
            'total_candidates': len(results),
            'successful_candidates': len(successful_results),
            'pareto_optimal_count': len(pareto_frontier),
            'objective_statistics': {}
        }
        
        # Analyze objective distributions
        for obj_name in self.objectives.keys():
            obj_values = []
            for result in successful_results:
                if 'objectives' in result and obj_name in result['objectives']:
                    obj_values.append(result['objectives'][obj_name])
            
            if obj_values:
                analysis['objective_statistics'][obj_name] = {
                    'mean': np.mean(obj_values),
                    'std': np.std(obj_values),
                    'min': np.min(obj_values),
                    'max': np.max(obj_values)
                }
        
        # Find best trade-off solutions
        if pareto_frontier:
            analysis['best_solutions'] = pareto_frontier[:3]  # Top 3 Pareto solutions
        
        return analysis
    
    def _save_results(self, results: List[Dict], filename: str):
        """Save optimization results."""
        results_path = self.results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {results_path}")
    
    def optimize(self, 
                 candidate_configs: List[Dict] = None,
                 evaluation_episodes: int = 5) -> Dict:
        """Main optimization interface."""
        
        if candidate_configs is None:
            # Generate default candidate configurations
            candidate_configs = self._generate_default_candidates()
        
        return self.optimize_multiobjective(candidate_configs, evaluation_episodes)
    
    def _generate_default_candidates(self) -> List[Dict]:
        """Generate default candidate configurations for multi-objective optimization."""
        
        candidates = []
        
        # Generate candidates with different trade-offs
        base_config = self._load_config(self.base_config_path)
        
        # Performance-focused candidates
        candidates.extend([
            {'learning_rate': 0.001, 'batch_size': 512, 'epochs': 50},  # High performance
            {'learning_rate': 0.0005, 'batch_size': 1024, 'epochs': 75},  # Very high performance
        ])
        
        # Efficiency-focused candidates  
        candidates.extend([
            {'learning_rate': 0.003, 'batch_size': 2048, 'epochs': 20},  # Fast training
            {'learning_rate': 0.005, 'batch_size': 4096, 'epochs': 15},  # Very fast training
        ])
        
        # Robustness-focused candidates
        candidates.extend([
            {'learning_rate': 0.0002, 'batch_size': 256, 'epochs': 100},  # Conservative, robust
            {'learning_rate': 0.001, 'batch_size': 512, 'epochs': 80},   # Balanced robust
        ])
        
        return candidates[:6]  # Limit to 6 candidates for reasonable runtime
