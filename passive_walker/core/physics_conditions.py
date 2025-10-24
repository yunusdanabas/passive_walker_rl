#!/usr/bin/env python3
"""
Diverse Physics Conditions for Robust Data Collection

This module implements diverse physics parameter variations to create robust
training data that generalizes across different physical conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PhysicsParameter(Enum):
    """Physics parameters that can be varied for robustness."""
    GRAVITY = "gravity"
    MASS = "mass"
    FRICTION = "friction"
    DAMPING = "damping"
    STIFFNESS = "stiffness"
    TIMESTEP = "timestep"


@dataclass
class PhysicsRange:
    """Range for a physics parameter variation."""
    min_value: float
    max_value: float
    default_value: float
    distribution: str = "uniform"  # "uniform", "normal", "log_uniform"


class PhysicsConditionGenerator:
    """Generates diverse physics conditions for robust data collection."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize physics condition generator.
        
        Args:
            seed: Random seed for reproducible conditions
        """
        self.rng = np.random.RandomState(seed)
        
        # Define physics parameter ranges
        self.parameter_ranges = self._define_parameter_ranges()
        
    def _define_parameter_ranges(self) -> Dict[PhysicsParameter, PhysicsRange]:
        """Define realistic ranges for physics parameters."""
        return {
            PhysicsParameter.GRAVITY: PhysicsRange(
                min_value=0.5,      # 50% of Earth gravity
                max_value=2.0,      # 200% of Earth gravity
                default_value=1.0,  # 100% of Earth gravity (scaling factor)
                distribution="uniform"
            ),
            PhysicsParameter.MASS: PhysicsRange(
                min_value=0.7,      # 70% of default mass
                max_value=1.5,      # 150% of default mass
                default_value=1.0,
                distribution="uniform"
            ),
            PhysicsParameter.FRICTION: PhysicsRange(
                min_value=0.3,      # Low friction
                max_value=1.2,      # High friction
                default_value=0.7,
                distribution="uniform"
            ),
            PhysicsParameter.DAMPING: PhysicsRange(
                min_value=0.1,      # Low damping
                max_value=2.0,      # High damping
                default_value=0.5,
                distribution="log_uniform"
            ),
            PhysicsParameter.STIFFNESS: PhysicsRange(
                min_value=0.5,      # Soft joints
                max_value=2.0,      # Stiff joints
                default_value=1.0,
                distribution="uniform"
            ),
            PhysicsParameter.TIMESTEP: PhysicsRange(
                min_value=0.005,    # 5ms timestep
                max_value=0.02,     # 20ms timestep
                default_value=0.01,
                distribution="uniform"
            )
        }
    
    def generate_condition(self, 
                          parameters: Optional[List[PhysicsParameter]] = None,
                          correlation_strength: float = 0.3) -> Dict[str, float]:
        """Generate a single physics condition.
        
        Args:
            parameters: List of parameters to vary (None for all)
            correlation_strength: Strength of parameter correlations (0-1)
            
        Returns:
            Dictionary mapping parameter names to values
        """
        if parameters is None:
            parameters = list(PhysicsParameter)
        
        condition = {}
        
        # Generate base values for each parameter
        for param in parameters:
            if param in self.parameter_ranges:
                range_def = self.parameter_ranges[param]
                condition[param.value] = self._sample_parameter(range_def)
        
        # Apply correlations if specified
        if correlation_strength > 0:
            condition = self._apply_correlations(condition, correlation_strength)
        
        return condition
    
    def generate_condition_batch(self, 
                                num_conditions: int,
                                parameters: Optional[List[PhysicsParameter]] = None,
                                correlation_strength: float = 0.3) -> List[Dict[str, float]]:
        """Generate a batch of diverse physics conditions.
        
        Args:
            num_conditions: Number of conditions to generate
            parameters: List of parameters to vary (None for all)
            correlation_strength: Strength of parameter correlations (0-1)
            
        Returns:
            List of physics condition dictionaries
        """
        conditions = []
        
        for _ in range(num_conditions):
            condition = self.generate_condition(parameters, correlation_strength)
            conditions.append(condition)
        
        return conditions
    
    def generate_curriculum_conditions(self, 
                                      stages: int = 4,
                                      conditions_per_stage: int = 10) -> Dict[int, List[Dict[str, float]]]:
        """Generate physics conditions for curriculum learning.
        
        Args:
            stages: Number of curriculum stages
            conditions_per_stage: Number of conditions per stage
            
        Returns:
            Dictionary mapping stage index to list of conditions
        """
        curriculum = {}
        
        for stage in range(stages):
            # Increase parameter variation range with stage
            variation_factor = (stage + 1) / stages
            
            # Select parameters based on stage
            if stage == 0:
                # Stage 1: Only basic parameters
                parameters = [PhysicsParameter.GRAVITY, PhysicsParameter.MASS]
            elif stage == 1:
                # Stage 2: Add friction and damping
                parameters = [PhysicsParameter.GRAVITY, PhysicsParameter.MASS, 
                            PhysicsParameter.FRICTION, PhysicsParameter.DAMPING]
            elif stage == 2:
                # Stage 3: Add stiffness
                parameters = [PhysicsParameter.GRAVITY, PhysicsParameter.MASS,
                            PhysicsParameter.FRICTION, PhysicsParameter.DAMPING,
                            PhysicsParameter.STIFFNESS]
            else:
                # Stage 4: All parameters including timestep
                parameters = list(PhysicsParameter)
            
            # Generate conditions for this stage
            stage_conditions = []
            for _ in range(conditions_per_stage):
                condition = self.generate_condition(parameters, correlation_strength=0.2)
                
                # Scale variation based on stage
                for param_name, value in condition.items():
                    param_enum = PhysicsParameter(param_name)
                    range_def = self.parameter_ranges[param_enum]
                    
                    # Scale the deviation from default
                    deviation = (value - range_def.default_value) * variation_factor
                    condition[param_name] = range_def.default_value + deviation
                
                stage_conditions.append(condition)
            
            curriculum[stage] = stage_conditions
        
        return curriculum
    
    def _sample_parameter(self, range_def: PhysicsRange) -> float:
        """Sample a parameter value from its range."""
        if range_def.distribution == "uniform":
            return self.rng.uniform(range_def.min_value, range_def.max_value)
        elif range_def.distribution == "normal":
            # Use normal distribution centered on default with std based on range
            std = (range_def.max_value - range_def.min_value) / 6  # 99.7% within range
            value = self.rng.normal(range_def.default_value, std)
            # Clamp to range
            return np.clip(value, range_def.min_value, range_def.max_value)
        elif range_def.distribution == "log_uniform":
            # Log-uniform distribution for parameters that vary over orders of magnitude
            log_min = np.log(range_def.min_value)
            log_max = np.log(range_def.max_value)
            log_value = self.rng.uniform(log_min, log_max)
            return np.exp(log_value)
        else:
            raise ValueError(f"Unknown distribution: {range_def.distribution}")
    
    def _apply_correlations(self, condition: Dict[str, float], strength: float) -> Dict[str, float]:
        """Apply correlations between parameters."""
        # Simple correlation: gravity affects mass scaling
        if "gravity" in condition and "mass" in condition:
            gravity_factor = condition["gravity"] / 9.81
            mass_factor = condition["mass"]
            
            # Correlate mass with gravity (heavier objects in higher gravity)
            correlated_mass = mass_factor * (1 + strength * (gravity_factor - 1))
            condition["mass"] = np.clip(correlated_mass, 
                                      self.parameter_ranges[PhysicsParameter.MASS].min_value,
                                      self.parameter_ranges[PhysicsParameter.MASS].max_value)
        
        # Damping correlates with stiffness
        if "damping" in condition and "stiffness" in condition:
            damping_factor = condition["damping"]
            stiffness_factor = condition["stiffness"]
            
            # Higher stiffness often requires higher damping
            correlated_damping = damping_factor * (1 + strength * (stiffness_factor - 1))
            condition["damping"] = np.clip(correlated_damping,
                                         self.parameter_ranges[PhysicsParameter.DAMPING].min_value,
                                         self.parameter_ranges[PhysicsParameter.DAMPING].max_value)
        
        return condition
    
    def get_default_condition(self) -> Dict[str, float]:
        """Get the default physics condition."""
        return {param.value: range_def.default_value 
                for param, range_def in self.parameter_ranges.items()}
    
    def validate_condition(self, condition: Dict[str, float]) -> bool:
        """Validate that a physics condition is within acceptable ranges."""
        for param_name, value in condition.items():
            try:
                param_enum = PhysicsParameter(param_name)
                range_def = self.parameter_ranges[param_enum]
                
                if not (range_def.min_value <= value <= range_def.max_value):
                    return False
            except (ValueError, KeyError):
                return False
        
        return True


class PhysicsConditionManager:
    """Manages physics conditions for data collection and training."""
    
    def __init__(self, generator: Optional[PhysicsConditionGenerator] = None):
        """Initialize physics condition manager.
        
        Args:
            generator: Physics condition generator (creates default if None)
        """
        self.generator = generator or PhysicsConditionGenerator()
        self.conditions_cache = {}
        
    def get_condition_for_episode(self, 
                                 episode_idx: int,
                                 total_episodes: int,
                                 curriculum_stage: int = 0) -> Dict[str, float]:
        """Get physics condition for a specific episode.
        
        Args:
            episode_idx: Current episode index
            total_episodes: Total number of episodes
            curriculum_stage: Current curriculum stage (0-3)
            
        Returns:
            Physics condition dictionary
        """
        # Use curriculum-based condition selection
        if curriculum_stage in self.conditions_cache:
            stage_conditions = self.conditions_cache[curriculum_stage]
        else:
            # Generate conditions for this stage
            curriculum = self.generator.generate_curriculum_conditions(
                stages=4, conditions_per_stage=20
            )
            stage_conditions = curriculum[curriculum_stage]
            self.conditions_cache[curriculum_stage] = stage_conditions
        
        # Select condition based on episode index
        condition_idx = episode_idx % len(stage_conditions)
        return stage_conditions[condition_idx]
    
    def apply_condition_to_env(self, env, condition: Dict[str, float]):
        """Apply physics condition to environment.
        
        Args:
            env: MuJoCo environment
            condition: Physics condition dictionary
        """
        # Apply gravity (as scaling factor)
        if "gravity" in condition:
            env.model.opt.gravity[2] = -9.81 * condition["gravity"]
        
        # Apply mass scaling
        if "mass" in condition:
            mass_scale = condition["mass"]
            for i in range(env.model.nbody):
                env.model.body_mass[i] *= mass_scale
        
        # Apply friction
        if "friction" in condition:
            friction_scale = condition["friction"]
            for i in range(env.model.ngeom):
                env.model.geom_friction[i] *= friction_scale
        
        # Apply damping
        if "damping" in condition:
            damping_scale = condition["damping"]
            for i in range(env.model.nv):
                env.model.dof_damping[i] *= damping_scale
        
        # Apply stiffness
        if "stiffness" in condition:
            stiffness_scale = condition["stiffness"]
            for i in range(env.model.nv):
                env.model.dof_armature[i] *= stiffness_scale
        
        # Apply timestep
        if "timestep" in condition:
            env.model.opt.timestep = condition["timestep"]
        
        # Forward the model to apply changes
        mujoco.mj_forward(env.model, env.data)
    
    def reset_to_default(self, env):
        """Reset environment to default physics condition."""
        default_condition = self.generator.get_default_condition()
        self.apply_condition_to_env(env, default_condition)


def create_physics_condition_manager(seed: Optional[int] = None) -> PhysicsConditionManager:
    """Create a physics condition manager with default settings."""
    generator = PhysicsConditionGenerator(seed=seed)
    return PhysicsConditionManager(generator)
