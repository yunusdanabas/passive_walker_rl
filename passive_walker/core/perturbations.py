"""
Perturbation System for Enhanced Data Collection

Implements various perturbation types to improve model robustness:
- Impulse perturbations (instantaneous forces/torques)
- Continuous push perturbations (sustained forces)
- Terrain changes (ramp angle variations)
- Mass distribution changes
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class PerturbationType(Enum):
    """Types of perturbations available."""
    IMPULSE_LATERAL = "impulse_lateral"
    IMPULSE_FRONTAL = "impulse_frontal"
    IMPULSE_TORSO = "impulse_torso"
    PUSH_LATERAL = "push_lateral"
    PUSH_FRONTAL = "push_frontal"
    TERRAIN_RAMP = "terrain_ramp"
    TERRAIN_FRICTION = "terrain_friction"
    MASS_TORSO = "mass_torso"
    MASS_LEGS = "mass_legs"


@dataclass
class PerturbationConfig:
    """Configuration for perturbation parameters."""
    # Force magnitudes (N)
    impulse_force_max: float = 50.0
    push_force_max: float = 20.0
    
    # Torque magnitudes (Nm)
    impulse_torque_max: float = 10.0
    
    # Terrain parameters
    ramp_angle_max: float = 15.0  # degrees
    friction_range: Tuple[float, float] = (0.3, 1.0)
    
    # Mass changes (percentage)
    mass_change_max: float = 0.2  # ±20%
    
    # Timing parameters
    min_interval: float = 1.0  # Minimum time between perturbations (s)
    max_interval: float = 5.0   # Maximum time between perturbations (s)
    
    # Duration for continuous perturbations
    push_duration_range: Tuple[float, float] = (0.5, 2.0)  # seconds


class PerturbationManager:
    """
    Manages perturbation injection during data collection.
    
    Supports multiple perturbation types with configurable timing,
    strength, and duration. Designed to work with MuJoCo environments.
    """
    
    def __init__(self, config: Optional[PerturbationConfig] = None):
        """
        Initialize perturbation manager.
        
        Args:
            config: Perturbation configuration parameters
        """
        self.config = config or PerturbationConfig()
        self.active_perturbations: Dict[str, Dict] = {}
        self.next_perturbation_time = 0.0
        self.last_perturbation_time = 0.0
        
    def update(self, env, timestep: float) -> Dict[str, any]:
        """
        Update perturbation state and apply active perturbations.
        
        Args:
            env: MuJoCo environment instance
            timestep: Current simulation time
            
        Returns:
            Dictionary with perturbation info for logging
        """
        perturbation_info = {
            'perturbations_applied': [],
            'active_perturbations': len(self.active_perturbations),
            'next_perturbation_time': self.next_perturbation_time
        }
        
        # Check if it's time for a new perturbation
        if timestep >= self.next_perturbation_time:
            self._schedule_next_perturbation(timestep)
        
        # Apply active perturbations
        for perturbation_id, perturbation_data in list(self.active_perturbations.items()):
            if self._apply_perturbation(env, perturbation_data):
                perturbation_info['perturbations_applied'].append(perturbation_id)
            else:
                # Remove completed perturbations
                del self.active_perturbations[perturbation_id]
        
        return perturbation_info
    
    def add_impulse(self, perturbation_type: PerturbationType, 
                   strength: float = 1.0, timestep: float = 0.0) -> str:
        """
        Add an impulse perturbation.
        
        Args:
            perturbation_type: Type of impulse perturbation
            strength: Strength multiplier (0.0 to 1.0)
            timestep: When to apply the perturbation
            
        Returns:
            Unique perturbation ID
        """
        perturbation_id = f"impulse_{len(self.active_perturbations)}"
        
        if perturbation_type == PerturbationType.IMPULSE_LATERAL:
            force = np.array([self.config.impulse_force_max * strength, 0, 0])
            torque = np.array([0, 0, 0])
        elif perturbation_type == PerturbationType.IMPULSE_FRONTAL:
            force = np.array([0, self.config.impulse_force_max * strength, 0])
            torque = np.array([0, 0, 0])
        elif perturbation_type == PerturbationType.IMPULSE_TORSO:
            force = np.array([0, 0, 0])
            torque = np.array([0, 0, self.config.impulse_torque_max * strength])
        else:
            raise ValueError(f"Invalid impulse perturbation type: {perturbation_type}")
        
        self.active_perturbations[perturbation_id] = {
            'type': 'impulse',
            'perturbation_type': perturbation_type,
            'force': force,
            'torque': torque,
            'start_time': timestep,
            'duration': 0.0,  # Instantaneous
            'strength': strength
        }
        
        return perturbation_id
    
    def add_continuous_push(self, perturbation_type: PerturbationType,
                           strength: float = 1.0, duration: Optional[float] = None) -> str:
        """
        Add a continuous push perturbation.
        
        Args:
            perturbation_type: Type of push perturbation
            strength: Strength multiplier (0.0 to 1.0)
            duration: Duration in seconds (if None, random within range)
            
        Returns:
            Unique perturbation ID
        """
        perturbation_id = f"push_{len(self.active_perturbations)}"
        
        if duration is None:
            duration = np.random.uniform(*self.config.push_duration_range)
        
        if perturbation_type == PerturbationType.PUSH_LATERAL:
            force = np.array([self.config.push_force_max * strength, 0, 0])
        elif perturbation_type == PerturbationType.PUSH_FRONTAL:
            force = np.array([0, self.config.push_force_max * strength, 0])
        else:
            raise ValueError(f"Invalid push perturbation type: {perturbation_type}")
        
        self.active_perturbations[perturbation_id] = {
            'type': 'push',
            'perturbation_type': perturbation_type,
            'force': force,
            'torque': np.array([0, 0, 0]),
            'start_time': 0.0,  # Will be set when applied
            'duration': duration,
            'strength': strength
        }
        
        return perturbation_id
    
    def add_terrain_change(self, perturbation_type: PerturbationType,
                          strength: float = 1.0) -> str:
        """
        Add a terrain change perturbation.
        
        Args:
            perturbation_type: Type of terrain change
            strength: Strength multiplier (0.0 to 1.0)
            
        Returns:
            Unique perturbation ID
        """
        perturbation_id = f"terrain_{len(self.active_perturbations)}"
        
        if perturbation_type == PerturbationType.TERRAIN_RAMP:
            angle_change = self.config.ramp_angle_max * strength * np.random.choice([-1, 1])
            terrain_params = {'ramp_angle': angle_change}
        elif perturbation_type == PerturbationType.TERRAIN_FRICTION:
            friction_change = np.random.uniform(*self.config.friction_range) * strength
            terrain_params = {'friction': friction_change}
        else:
            raise ValueError(f"Invalid terrain perturbation type: {perturbation_type}")
        
        self.active_perturbations[perturbation_id] = {
            'type': 'terrain',
            'perturbation_type': perturbation_type,
            'terrain_params': terrain_params,
            'start_time': 0.0,
            'duration': float('inf'),  # Permanent change
            'strength': strength
        }
        
        return perturbation_id
    
    def add_mass_change(self, perturbation_type: PerturbationType,
                       strength: float = 1.0) -> str:
        """
        Add a mass distribution change perturbation.
        
        Args:
            perturbation_type: Type of mass change
            strength: Strength multiplier (0.0 to 1.0)
            
        Returns:
            Unique perturbation ID
        """
        perturbation_id = f"mass_{len(self.active_perturbations)}"
        
        mass_change = self.config.mass_change_max * strength * np.random.choice([-1, 1])
        
        if perturbation_type == PerturbationType.MASS_TORSO:
            mass_params = {'torso_mass_change': mass_change}
        elif perturbation_type == PerturbationType.MASS_LEGS:
            mass_params = {'leg_mass_change': mass_change}
        else:
            raise ValueError(f"Invalid mass perturbation type: {perturbation_type}")
        
        self.active_perturbations[perturbation_id] = {
            'type': 'mass',
            'perturbation_type': perturbation_type,
            'mass_params': mass_params,
            'start_time': 0.0,
            'duration': float('inf'),  # Permanent change
            'strength': strength
        }
        
        return perturbation_id
    
    def _schedule_next_perturbation(self, timestep: float):
        """Schedule the next perturbation time."""
        interval = np.random.uniform(self.config.min_interval, self.config.max_interval)
        self.next_perturbation_time = timestep + interval
        self.last_perturbation_time = timestep
    
    def _apply_perturbation(self, env, perturbation_data: Dict) -> bool:
        """
        Apply a single perturbation to the environment.
        
        Args:
            env: MuJoCo environment instance
            perturbation_data: Perturbation configuration
            
        Returns:
            True if perturbation was applied, False if completed
        """
        perturbation_type = perturbation_data['type']
        
        if perturbation_type == 'impulse':
            return self._apply_impulse(env, perturbation_data)
        elif perturbation_type == 'push':
            return self._apply_push(env, perturbation_data)
        elif perturbation_type == 'terrain':
            return self._apply_terrain_change(env, perturbation_data)
        elif perturbation_type == 'mass':
            return self._apply_mass_change(env, perturbation_data)
        else:
            return False
    
    def _apply_impulse(self, env, perturbation_data: Dict) -> bool:
        """Apply impulse perturbation (instantaneous force/torque)."""
        # Apply force to torso
        force = perturbation_data['force']
        torque = perturbation_data['torque']
        
        # Get torso body ID (assuming it's named 'torso' or similar)
        torso_id = self._get_torso_body_id(env)
        if torso_id is not None:
            env.data.xfrc_applied[torso_id, :3] = force
            env.data.xfrc_applied[torso_id, 3:] = torque
        
        # Impulse is instantaneous, so remove immediately
        return False
    
    def _apply_push(self, env, perturbation_data: Dict) -> bool:
        """Apply continuous push perturbation."""
        current_time = env.data.time
        
        # Initialize start time if not set
        if perturbation_data['start_time'] == 0.0:
            perturbation_data['start_time'] = current_time
        
        # Check if push duration has expired
        elapsed = current_time - perturbation_data['start_time']
        if elapsed >= perturbation_data['duration']:
            return False
        
        # Apply continuous force
        force = perturbation_data['force']
        torso_id = self._get_torso_body_id(env)
        if torso_id is not None:
            env.data.xfrc_applied[torso_id, :3] = force
        
        return True
    
    def _apply_terrain_change(self, env, perturbation_data: Dict) -> bool:
        """Apply terrain change perturbation."""
        terrain_params = perturbation_data['terrain_params']
        
        # Apply terrain changes to environment
        if 'ramp_angle' in terrain_params:
            self._set_ramp_angle(env, terrain_params['ramp_angle'])
        if 'friction' in terrain_params:
            self._set_friction(env, terrain_params['friction'])
        
        # Terrain changes are permanent
        return True
    
    def _apply_mass_change(self, env, perturbation_data: Dict) -> bool:
        """Apply mass change perturbation."""
        mass_params = perturbation_data['mass_params']
        
        # Apply mass changes to environment
        if 'torso_mass_change' in mass_params:
            self._change_torso_mass(env, mass_params['torso_mass_change'])
        if 'leg_mass_change' in mass_params:
            self._change_leg_mass(env, mass_params['leg_mass_change'])
        
        # Mass changes are permanent
        return True
    
    def _get_torso_body_id(self, env) -> Optional[int]:
        """Get torso body ID from environment."""
        try:
            # Try common torso body names
            for name in ['torso', 'pelvis', 'body']:
                if hasattr(env.model, 'body') and name in env.model.body_names:
                    return env.model.body(name).id
        except:
            pass
        return None
    
    def _set_ramp_angle(self, env, angle: float):
        """Set ramp angle in environment."""
        try:
            # This would need to be implemented based on environment structure
            # For now, we'll store it for the environment to use
            if not hasattr(env, '_perturbation_ramp_angle'):
                env._perturbation_ramp_angle = 0.0
            env._perturbation_ramp_angle += angle
        except:
            pass
    
    def _set_friction(self, env, friction: float):
        """Set friction in environment."""
        try:
            # This would need to be implemented based on environment structure
            if not hasattr(env, '_perturbation_friction'):
                env._perturbation_friction = 1.0
            env._perturbation_friction = friction
        except:
            pass
    
    def _change_torso_mass(self, env, mass_change: float):
        """Change torso mass in environment."""
        try:
            # This would need to be implemented based on environment structure
            if not hasattr(env, '_perturbation_torso_mass_change'):
                env._perturbation_torso_mass_change = 0.0
            env._perturbation_torso_mass_change += mass_change
        except:
            pass
    
    def _change_leg_mass(self, env, mass_change: float):
        """Change leg mass in environment."""
        try:
            # This would need to be implemented based on environment structure
            if not hasattr(env, '_perturbation_leg_mass_change'):
                env._perturbation_leg_mass_change = 0.0
            env._perturbation_leg_mass_change += mass_change
        except:
            pass
    
    def reset(self):
        """Reset perturbation manager state."""
        self.active_perturbations.clear()
        self.next_perturbation_time = 0.0
        self.last_perturbation_time = 0.0
    
    def get_stats(self) -> Dict[str, any]:
        """Get perturbation statistics."""
        return {
            'active_perturbations': len(self.active_perturbations),
            'next_perturbation_time': self.next_perturbation_time,
            'last_perturbation_time': self.last_perturbation_time,
            'perturbation_types': [p['perturbation_type'].value for p in self.active_perturbations.values()]
        }


def create_perturbation_manager(mode: str = "random", 
                               strength: float = 1.0,
                               config: Optional[PerturbationConfig] = None) -> PerturbationManager:
    """
    Create a perturbation manager with predefined settings.
    
    Args:
        mode: Perturbation mode ("none", "random", "scheduled", "curriculum")
        strength: Overall perturbation strength multiplier
        config: Custom perturbation configuration
        
    Returns:
        Configured PerturbationManager instance
    """
    manager = PerturbationManager(config)
    
    if mode == "none":
        # No perturbations
        pass
    elif mode == "random":
        # Random perturbations with random timing
        pass  # Default behavior
    elif mode == "scheduled":
        # Scheduled perturbations every 2 seconds
        manager.config.min_interval = 2.0
        manager.config.max_interval = 2.0
    elif mode == "curriculum":
        # Progressive difficulty
        manager.config.impulse_force_max *= strength
        manager.config.push_force_max *= strength
        manager.config.min_interval = max(0.5, 3.0 - strength * 2.0)
        manager.config.max_interval = max(1.0, 5.0 - strength * 2.0)
    
    return manager

