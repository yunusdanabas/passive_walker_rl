"""
Environment Enhancements for Training

Online perturbations, adaptive randomization, and reward curriculum.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple
import mujoco


class OnlinePerturbationManager:
    """
    Manages online perturbations during training.
    
    Applies disturbances to the environment during episodes.
    """
    
    def __init__(self, 
                 impulse_probability: float = 0.1,
                 impulse_magnitude: float = 50.0,
                 continuous_push_probability: float = 0.05,
                 continuous_push_magnitude: float = 10.0,
                 terrain_change_probability: float = 0.02):
        """
        Initialize perturbation manager.
        
        Args:
            impulse_probability: Probability of impulse per step
            impulse_magnitude: Magnitude of impulse forces
            continuous_push_probability: Probability of continuous push per step
            continuous_push_magnitude: Magnitude of continuous push
            terrain_change_probability: Probability of terrain change per step
        """
        self.impulse_probability = impulse_probability
        self.impulse_magnitude = impulse_magnitude
        self.continuous_push_probability = continuous_push_probability
        self.continuous_push_magnitude = continuous_push_magnitude
        self.terrain_change_probability = terrain_change_probability
        
        # State tracking
        self.active_push = None
        self.push_duration = 0
        self.max_push_duration = 10  # steps
        
        # Terrain state
        self.original_ramp_deg = None
        self.terrain_change_active = False
        self.terrain_change_duration = 0
        self.max_terrain_duration = 50  # steps
    
    def apply_perturbation(self, model: mujoco.MjModel, data: mujoco.MjData, 
                         step: int, rng: np.random.RandomState) -> Dict[str, Any]:
        """
        Apply perturbations to the environment.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            step: Current step
            rng: Random number generator
            
        Returns:
            Dictionary of perturbation info
        """
        perturbation_info = {
            "impulse_applied": False,
            "continuous_push_active": False,
            "terrain_changed": False
        }
        
        # Apply impulse perturbation
        if rng.random() < self.impulse_probability:
            self._apply_impulse(model, data, rng)
            perturbation_info["impulse_applied"] = True
        
        # Apply continuous push
        if rng.random() < self.continuous_push_probability:
            self._start_continuous_push(rng)
        
        if self.active_push is not None:
            self._apply_continuous_push(model, data)
            perturbation_info["continuous_push_active"] = True
            
            # End push if duration exceeded
            if self.push_duration >= self.max_push_duration:
                self.active_push = None
                self.push_duration = 0
        
        # Apply terrain change
        if rng.random() < self.terrain_change_probability:
            self._apply_terrain_change(model, rng)
            perturbation_info["terrain_changed"] = True
        
        # Reset terrain change if duration exceeded
        if self.terrain_change_active:
            self.terrain_change_duration += 1
            if self.terrain_change_duration >= self.max_terrain_duration:
                self._reset_terrain(model)
                self.terrain_change_active = False
                self.terrain_change_duration = 0
        
        return perturbation_info
    
    def _apply_impulse(self, model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.RandomState):
        """Apply impulse force to torso."""
        # Get torso body ID (assuming it's body 0)
        torso_id = 0
        
        # Random direction for impulse
        direction = rng.uniform(-1, 1, 3)
        direction[1] = 0  # No vertical impulse
        direction = direction / np.linalg.norm(direction)
        
        # Apply impulse
        impulse = direction * self.impulse_magnitude
        data.xfrc_applied[torso_id, :3] = impulse
    
    def _start_continuous_push(self, rng: np.random.RandomState):
        """Start continuous push perturbation."""
        if self.active_push is None:
            # Random direction for push
            direction = rng.uniform(-1, 1, 3)
            direction[1] = 0  # No vertical push
            direction = direction / np.linalg.norm(direction)
            
            self.active_push = direction * self.continuous_push_magnitude
            self.push_duration = 0
    
    def _apply_continuous_push(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Apply continuous push force."""
        if self.active_push is not None:
            torso_id = 0
            data.xfrc_applied[torso_id, :3] = self.active_push
            self.push_duration += 1
    
    def _apply_terrain_change(self, model: mujoco.MjModel, rng: np.random.RandomState):
        """Apply terrain change (ramp angle modification)."""
        if not self.terrain_change_active:
            # Store original ramp angle
            if self.original_ramp_deg is None:
                self.original_ramp_deg = model.opt.gravity[0] / 9.81
            
            # Random terrain change
            terrain_change = rng.uniform(-5.0, 5.0)  # degrees
            new_ramp_deg = self.original_ramp_deg + terrain_change
            
            # Apply terrain change
            tilt = np.deg2rad(new_ramp_deg)
            model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]
            
            self.terrain_change_active = True
            self.terrain_change_duration = 0
    
    def _reset_terrain(self, model: mujoco.MjModel):
        """Reset terrain to original state."""
        if self.original_ramp_deg is not None:
            tilt = np.deg2rad(self.original_ramp_deg)
            model.opt.gravity[:] = [9.81 * np.sin(tilt), 0.0, -9.81 * np.cos(tilt)]


class AdaptiveRandomizationManager:
    """
    Manages adaptive domain randomization based on performance.
    
    Adjusts randomization strength based on training progress.
    """
    
    def __init__(self,
                 base_randomization_strength: float = 0.5,
                 adaptation_rate: float = 0.1,
                 min_strength: float = 0.1,
                 max_strength: float = 1.0):
        """
        Initialize adaptive randomization manager.
        
        Args:
            base_randomization_strength: Base randomization strength
            adaptation_rate: Rate of adaptation
            min_strength: Minimum randomization strength
            max_strength: Maximum randomization strength
        """
        self.base_strength = base_randomization_strength
        self.current_strength = base_randomization_strength
        self.adaptation_rate = adaptation_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        # Performance tracking
        self.performance_history = []
        self.randomization_history = []
    
    def update_performance(self, performance_metric: float):
        """
        Update performance metric and adjust randomization.
        
        Args:
            performance_metric: Current performance (higher is better)
        """
        self.performance_history.append(performance_metric)
        self.randomization_history.append(self.current_strength)
        
        if len(self.performance_history) >= 2:
            # Compare with previous performance
            if performance_metric > self.performance_history[-2]:
                # Performance improved, increase randomization
                self.current_strength *= (1 + self.adaptation_rate)
            else:
                # Performance decreased, reduce randomization
                self.current_strength *= (1 - self.adaptation_rate)
            
            # Clamp strength
            self.current_strength = np.clip(
                self.current_strength,
                self.min_strength,
                self.max_strength
            )
    
    def get_randomization_params(self) -> Dict[str, float]:
        """
        Get current randomization parameters.
        
        Returns:
            Dictionary of randomization parameters
        """
        return {
            "mass_jitter": 0.05 * self.current_strength,
            "ramp_jitter": 2.0 * self.current_strength,
            "friction_range": 0.4 * self.current_strength,
            "strength": self.current_strength
        }
    
    def reset(self):
        """Reset to base randomization strength."""
        self.current_strength = self.base_strength
        self.performance_history.clear()
        self.randomization_history.clear()


class RewardCurriculumManager:
    """
    Manages reward curriculum for progressive training.
    
    Starts with simple rewards and transitions to complex rewards.
    """
    
    def __init__(self,
                 total_episodes: int,
                 transition_episodes: int = 1000,
                 start_mode: str = "fsm",
                 end_mode: str = "research"):
        """
        Initialize reward curriculum manager.
        
        Args:
            total_episodes: Total training episodes
            transition_episodes: Episodes for transition
            start_mode: Starting reward mode
            end_mode: Ending reward mode
        """
        self.total_episodes = total_episodes
        self.transition_episodes = transition_episodes
        self.start_mode = start_mode
        self.end_mode = end_mode
        
        self.current_episode = 0
    
    def get_current_mode(self, episode: int) -> str:
        """
        Get current reward mode based on episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Current reward mode
        """
        self.current_episode = episode
        
        if episode < self.transition_episodes:
            # Transition phase
            progress = episode / self.transition_episodes
            
            # Interpolate between modes
            if progress < 0.5:
                return self.start_mode
            else:
                # Transition to end mode
                return self.end_mode
        else:
            # Final phase
            return self.end_mode
    
    def get_reward_weights(self, episode: int) -> Dict[str, float]:
        """
        Get current reward weights based on episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Dictionary of reward weights
        """
        if episode < self.transition_episodes:
            progress = episode / self.transition_episodes
            
            # Start with simple weights, transition to complex
            if progress < 0.5:
                # Simple weights (FSM-like)
                return {
                    "w_dx": 2.0,
                    "w_pitch": 0.1,
                    "w_ctrl": 0.0005,
                    "w_alive": 0.20,
                    "w_velocity": 0.0,
                    "w_symmetry": 0.0,
                    "w_foot_clear": 0.0,
                    "w_smooth": 0.0
                }
            else:
                # Transition to complex weights
                transition_progress = (progress - 0.5) * 2
                return {
                    "w_dx": 2.0 - transition_progress,
                    "w_pitch": 0.1,
                    "w_ctrl": 0.0005 + transition_progress * 0.001,
                    "w_alive": 0.20 + transition_progress * 0.30,
                    "w_velocity": transition_progress * 0.3,
                    "w_symmetry": transition_progress * 0.2,
                    "w_foot_clear": transition_progress * 0.2,
                    "w_smooth": transition_progress * 0.1
                }
        else:
            # Complex weights (research mode)
            return {
                "w_dx": 1.0,
                "w_pitch": 0.1,
                "w_ctrl": 0.0015,
                "w_alive": 0.5,
                "w_velocity": 0.3,
                "w_symmetry": 0.2,
                "w_foot_clear": 0.2,
                "w_smooth": 0.1
            }
    
    def update_episode(self, episode: int):
        """Update current episode."""
        self.current_episode = episode


def create_online_perturbation_manager(**kwargs) -> OnlinePerturbationManager:
    """Create online perturbation manager with default settings."""
    return OnlinePerturbationManager(**kwargs)


def create_adaptive_randomization_manager(**kwargs) -> AdaptiveRandomizationManager:
    """Create adaptive randomization manager with default settings."""
    return AdaptiveRandomizationManager(**kwargs)


def create_reward_curriculum_manager(total_episodes: int, **kwargs) -> RewardCurriculumManager:
    """Create reward curriculum manager with default settings."""
    return RewardCurriculumManager(total_episodes, **kwargs)
