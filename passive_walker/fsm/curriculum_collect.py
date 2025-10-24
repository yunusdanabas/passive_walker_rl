#!/usr/bin/env python3
"""
Curriculum Data Collection for Passive Walker FSM

This script implements progressive difficulty data collection across 4 stages:
- Stage 1: Basic walking (no perturbations)
- Stage 2: Light perturbations (impulse, push)
- Stage 3: Medium perturbations (terrain, mass)
- Stage 4: Heavy perturbations (combined)

Each stage builds upon the previous one, creating a curriculum of increasing complexity.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mujoco

from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.perturbations import PerturbationManager, PerturbationType
from passive_walker.core.physics_conditions import PhysicsConditionManager, create_physics_condition_manager
from passive_walker.fsm.collect import collect


class CurriculumCollector:
    """Progressive difficulty data collection across multiple stages."""
    
    def __init__(self, output_dir: str, base_config: Dict = None):
        """Initialize curriculum collector.
        
        Args:
            output_dir: Directory to save curriculum data
            base_config: Base configuration for data collection
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default base configuration
        self.base_config = base_config or {
            "episodes": 50,
            "max_steps": 1000,
            "dt": 0.01,
            "use_gui": False,
            "save_frequency": 10
        }
        
        # Initialize physics condition manager
        self.physics_manager = create_physics_condition_manager(seed=42)
        
        # Stage definitions with increasing difficulty
        self.stages = self._define_stages()
        
    def _define_stages(self) -> List[Dict]:
        """Define the 4 curriculum stages with increasing perturbation complexity."""
        return [
            {
                "name": "stage1_basic",
                "description": "Basic walking with no perturbations",
                "perturbations": {
                    "enabled": False,
                    "types": [],
                    "probabilities": {},
                    "intensities": {}
                },
                "episodes": self.base_config["episodes"],
                "difficulty": 1.0
            },
            {
                "name": "stage2_light",
                "description": "Light perturbations (impulse, push)",
                "perturbations": {
                    "enabled": True,
                    "types": [PerturbationType.IMPULSE_LATERAL, PerturbationType.PUSH_LATERAL],
                    "probabilities": {
                        PerturbationType.IMPULSE_LATERAL: 0.3,
                        PerturbationType.PUSH_LATERAL: 0.2
                    },
                    "intensities": {
                        PerturbationType.IMPULSE_LATERAL: 0.5,  # 50% of max intensity
                        PerturbationType.PUSH_LATERAL: 0.4
                    }
                },
                "episodes": self.base_config["episodes"],
                "difficulty": 2.0
            },
            {
                "name": "stage3_medium",
                "description": "Medium perturbations (terrain, mass)",
                "perturbations": {
                    "enabled": True,
                    "types": [
                        PerturbationType.IMPULSE_LATERAL, 
                        PerturbationType.PUSH_LATERAL,
                        PerturbationType.TERRAIN_RAMP,
                        PerturbationType.MASS_TORSO
                    ],
                    "probabilities": {
                        PerturbationType.IMPULSE_LATERAL: 0.4,
                        PerturbationType.PUSH_LATERAL: 0.3,
                        PerturbationType.TERRAIN_RAMP: 0.3,
                        PerturbationType.MASS_TORSO: 0.2
                    },
                    "intensities": {
                        PerturbationType.IMPULSE_LATERAL: 0.7,
                        PerturbationType.PUSH_LATERAL: 0.6,
                        PerturbationType.TERRAIN_RAMP: 0.6,
                        PerturbationType.MASS_TORSO: 0.5
                    }
                },
                "episodes": self.base_config["episodes"],
                "difficulty": 3.0
            },
            {
                "name": "stage4_heavy",
                "description": "Heavy perturbations (combined, high intensity)",
                "perturbations": {
                    "enabled": True,
                    "types": [
                        PerturbationType.IMPULSE_LATERAL,
                        PerturbationType.PUSH_LATERAL,
                        PerturbationType.TERRAIN_RAMP,
                        PerturbationType.MASS_TORSO
                    ],
                    "probabilities": {
                        PerturbationType.IMPULSE_LATERAL: 0.5,
                        PerturbationType.PUSH_LATERAL: 0.4,
                        PerturbationType.TERRAIN_RAMP: 0.4,
                        PerturbationType.MASS_TORSO: 0.3
                    },
                    "intensities": {
                        PerturbationType.IMPULSE_LATERAL: 1.0,  # Full intensity
                        PerturbationType.PUSH_LATERAL: 0.9,
                        PerturbationType.TERRAIN_RAMP: 0.8,
                        PerturbationType.MASS_TORSO: 0.7
                    }
                },
                "episodes": self.base_config["episodes"],
                "difficulty": 4.0
            }
        ]
    
    def collect_curriculum(self, start_stage: int = 0, end_stage: int = None) -> Dict:
        """Collect data across curriculum stages.
        
        Args:
            start_stage: Starting stage index (0-based)
            end_stage: Ending stage index (inclusive, None for all stages)
            
        Returns:
            Dictionary with collection results and metadata
        """
        if end_stage is None:
            end_stage = len(self.stages) - 1
            
        results = {
            "curriculum_info": {
                "total_stages": len(self.stages),
                "start_stage": start_stage,
                "end_stage": end_stage,
                "base_config": self.base_config
            },
            "stage_results": {},
            "collection_time": time.time()
        }
        
        print(f"Starting curriculum collection: Stages {start_stage}-{end_stage}")
        print(f"Total stages: {len(self.stages)}")
        
        for stage_idx in range(start_stage, end_stage + 1):
            if stage_idx >= len(self.stages):
                break
                
            stage = self.stages[stage_idx]
            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}: {stage['name'].upper()}")
            print(f"Description: {stage['description']}")
            print(f"Difficulty: {stage['difficulty']}/4.0")
            print(f"Episodes: {stage['episodes']}")
            print(f"{'='*60}")
            
            # Collect data for this stage
            stage_result = self._collect_stage(stage_idx, stage)
            results["stage_results"][stage["name"]] = stage_result
            
            # Save intermediate results
            self._save_intermediate_results(results, stage_idx)
            
        # Save final results
        self._save_final_results(results)
        
        print(f"\n{'='*60}")
        print("CURRICULUM COLLECTION COMPLETED")
        print(f"{'='*60}")
        self._print_summary(results)
        
        return results
    
    def _collect_stage(self, stage_idx: int, stage: Dict) -> Dict:
        """Collect data for a single curriculum stage."""
        stage_start_time = time.time()
        
        # Create stage-specific output directory
        stage_dir = self.output_dir / stage["name"]
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare collection configuration
        config = self.base_config.copy()
        config.update({
            "episodes": stage["episodes"],
            "output_dir": str(stage_dir),
            "perturbations": stage["perturbations"]
        })
        
        # Collect data using the enhanced FSM collection
        try:
            collection_result = collect(
                episodes=config["episodes"],
                duration_sec=config["max_steps"] * config["dt"],  # Convert steps to duration
                outdir=config["output_dir"],
                seed=42,
                perturbation_mode="enabled" if config["perturbations"]["enabled"] else "none",
                perturbation_strength=1.0,
                perturbation_freq=0.5
            )
            
            stage_result = {
                "stage_info": stage,
                "collection_result": collection_result,
                "collection_time": time.time() - stage_start_time,
                "success": True
            }
            
        except Exception as e:
            print(f"Error in stage {stage_idx + 1}: {e}")
            stage_result = {
                "stage_info": stage,
                "error": str(e),
                "collection_time": time.time() - stage_start_time,
                "success": False
            }
        
        return stage_result
    
    def _save_intermediate_results(self, results: Dict, stage_idx: int):
        """Save intermediate results after each stage."""
        intermediate_file = self.output_dir / f"curriculum_intermediate_stage_{stage_idx + 1}.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(intermediate_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Intermediate results saved: {intermediate_file}")
    
    def _save_final_results(self, results: Dict):
        """Save final curriculum results."""
        final_file = self.output_dir / "curriculum_final_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(final_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Final results saved: {final_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _print_summary(self, results: Dict):
        """Print a summary of the curriculum collection results."""
        print(f"Total collection time: {time.time() - results['collection_time']:.2f} seconds")
        print(f"Stages completed: {len(results['stage_results'])}")
        
        for stage_name, stage_result in results["stage_results"].items():
            if stage_result["success"]:
                print(f"  {stage_name}: SUCCESS ({stage_result['collection_time']:.2f}s)")
            else:
                print(f"  {stage_name}: FAILED ({stage_result['collection_time']:.2f}s)")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("Files created:")
        for file_path in self.output_dir.rglob("*.json"):
            print(f"  {file_path.relative_to(self.output_dir)}")


def main():
    """Main function for curriculum collection."""
    parser = argparse.ArgumentParser(description="Curriculum Data Collection for Passive Walker FSM")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="experiments/data/curriculum",
                      help="Directory to save curriculum data")
    
    # Stage selection
    parser.add_argument("--start-stage", type=int, default=0,
                      help="Starting stage index (0-based)")
    parser.add_argument("--end-stage", type=int, default=None,
                      help="Ending stage index (inclusive, None for all stages)")
    
    # Collection parameters
    parser.add_argument("--episodes", type=int, default=50,
                      help="Number of episodes per stage")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="Maximum steps per episode")
    parser.add_argument("--dt", type=float, default=0.01,
                      help="Simulation timestep")
    parser.add_argument("--use-gui", action="store_true",
                      help="Enable GUI visualization")
    parser.add_argument("--save-frequency", type=int, default=10,
                      help="Save frequency for intermediate results")
    
    args = parser.parse_args()
    
    # Create base configuration
    base_config = {
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "dt": args.dt,
        "use_gui": args.use_gui,
        "save_frequency": args.save_frequency
    }
    
    # Initialize curriculum collector
    collector = CurriculumCollector(args.output_dir, base_config)
    
    # Collect curriculum data
    results = collector.collect_curriculum(args.start_stage, args.end_stage)
    
    return results


if __name__ == "__main__":
    main()
