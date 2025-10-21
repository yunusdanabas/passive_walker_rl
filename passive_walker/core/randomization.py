"""
Advanced Domain Randomization for Passive Walker

Provides comprehensive randomization utilities for robust policy training.
Reimplemented from legacy with modern APIs.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RandomizationConfig:
    """Configuration for domain randomization parameters."""
    ramp_deg_min: float = 8.0
    ramp_deg_max: float = 12.0
    friction_min: float = 0.6
    friction_max: float = 1.0
    mass_jitter: float = 0.05
    damping_jitter: float = 0.1
    enable_damping_randomization: bool = False
    actuator_gain_jitter: float = 0.05
    enable_actuator_randomization: bool = False
    enable_temporal_randomization: bool = False
    temporal_update_interval: int = 500


class DomainRandomizer:
    """Advanced domain randomization for the Passive Walker environment."""
    
    def __init__(self, config: RandomizationConfig, model, rng: np.random.RandomState):
        self.config = config
        self.model = model
        self.rng = rng
        self._cache_original_parameters()
    
    def _cache_original_parameters(self):
        """Cache original model parameters for randomization."""
        import mujoco
        self._original_masses = self.model.body_mass.copy()
        self._original_dof_damping = self.model.dof_damping.copy()
        self._original_actuator_gainprm = self.model.actuator_gainprm.copy()
        self.b_torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    
    def randomize_all(self, base_ramp_deg: float, base_friction: float) -> Tuple[float, float]:
        """Apply all enabled randomizations."""
        ramp_deg = self.rng.uniform(self.config.ramp_deg_min, self.config.ramp_deg_max)
        friction = self.rng.uniform(self.config.friction_min, self.config.friction_max)
        
        self._randomize_mass()
        if self.config.enable_damping_randomization:
            self._randomize_damping()
        if self.config.enable_actuator_randomization:
            self._randomize_actuators()
        
        return ramp_deg, friction
    
    def _randomize_mass(self):
        """Randomize body masses."""
        jitter = self.config.mass_jitter
        scale = self.rng.uniform(1.0 - jitter, 1.0 + jitter)
        self.model.body_mass[self.b_torso] = self._original_masses[self.b_torso] * scale
    
    def _randomize_damping(self):
        """Randomize joint damping coefficients."""
        jitter = self.config.damping_jitter
        for i in range(len(self.model.dof_damping)):
            if self._original_dof_damping[i] > 0:
                scale = self.rng.uniform(1.0 - jitter, 1.0 + jitter)
                self.model.dof_damping[i] = self._original_dof_damping[i] * scale
    
    def _randomize_actuators(self):
        """Randomize actuator parameters."""
        jitter = self.config.actuator_gain_jitter
        for i in range(self.model.nu):
            if self._original_actuator_gainprm[i, 0] != 0:
                scale = self.rng.uniform(1.0 - jitter, 1.0 + jitter)
                self.model.actuator_gainprm[i, 0] = self._original_actuator_gainprm[i, 0] * scale


# Predefined randomization profiles
RANDOMIZATION_PROFILES = {
    "none": RandomizationConfig(ramp_deg_min=10.0, ramp_deg_max=10.0, friction_min=0.9, friction_max=0.9, mass_jitter=0.0),
    "basic": RandomizationConfig(ramp_deg_min=9.0, ramp_deg_max=11.0, friction_min=0.8, friction_max=1.0, mass_jitter=0.05),
    "moderate": RandomizationConfig(ramp_deg_min=8.0, ramp_deg_max=12.0, friction_min=0.6, friction_max=1.0, mass_jitter=0.08, enable_damping_randomization=True),
    "aggressive": RandomizationConfig(ramp_deg_min=7.0, ramp_deg_max=13.0, friction_min=0.5, friction_max=1.1, mass_jitter=0.10, enable_damping_randomization=True, enable_actuator_randomization=True),
    "temporal": RandomizationConfig(ramp_deg_min=8.0, ramp_deg_max=12.0, friction_min=0.6, friction_max=1.0, mass_jitter=0.05, enable_damping_randomization=True, enable_actuator_randomization=True, enable_temporal_randomization=True),
}


def get_randomization_config(profile: str = "basic") -> RandomizationConfig:
    """Get a predefined randomization configuration."""
    if profile not in RANDOMIZATION_PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Choose from {list(RANDOMIZATION_PROFILES.keys())}")
    return RANDOMIZATION_PROFILES[profile]

