"""
Controller modules for the passive walker.

Provides PD control and FSM-based state machine for generating desired joint positions.
"""

import numpy as np
from typing import Tuple


class PDController:
    """Proportional-Derivative controller for joint actuation."""

    def __init__(self, control_cfg):
        """Initialize PD controller with gains and limits."""
        self.kp = np.array(control_cfg.kp, dtype=np.float32)
        self.kv = np.array(control_cfg.kv, dtype=np.float32)
        self.umin = np.array(control_cfg.umin, dtype=np.float32)
        self.umax = np.array(control_cfg.umax, dtype=np.float32)
        self.joint_ranges = np.array(control_cfg.joint_ranges, dtype=np.float32)

    def denorm(self, joint_idx: int, a_norm: float) -> float:
        """Convert normalized action [-1, 1] to joint position within range."""
        joint_min, joint_max = self.joint_ranges[joint_idx]
        return joint_min + (a_norm + 1.0) * 0.5 * (joint_max - joint_min)

    def step(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray) -> np.ndarray:
        """Compute PD torques: u = kp * (q_des - q) - kv * qd"""
        u = self.kp * (q_des - q) - self.kv * qd
        return np.clip(u, self.umin, self.umax)


class FSMStateMachine:
    """Finite State Machine for generating walking gaits."""

    def __init__(self):
        """Initialize FSM with default parameters."""
        # State variables
        self.hip_state = 0  # 0: left leg swing, 1: right leg swing
        self.knee_states = [0, 0]  # 0: stance (extended), 1: swing (retracted)

        # Timing variables
        self.hip_timer = 0.0
        self.knee_timers = [0.0, 0.0]

        # Gait parameters
        self.hip_period = 1.0  # seconds
        self.knee_period = 0.5  # seconds
        self.knee_retract_time = 0.1  # seconds to retract

    def reset(self):
        """Reset FSM to initial state."""
        self.hip_state = 0
        self.knee_states = [0, 0]
        self.hip_timer = 0.0
        self.knee_timers = [0.0, 0.0]

    def update(self, mj_data, mj_model):
        """Update FSM states based on simulation data."""
        dt = mj_model.opt.timestep
        self.hip_timer += dt
        self.knee_timers[0] += dt
        self.knee_timers[1] += dt

        # Hip state machine: alternate between legs
        if self.hip_timer >= self.hip_period:
            self.hip_state = 1 - self.hip_state
            self.hip_timer = 0.0

        # Knee state machine: stance/swing based on hip state
        for i in range(2):
            if self.hip_state == i:  # This leg is swinging
                if self.knee_states[i] == 0:  # Currently in stance
                    self.knee_states[i] = 1  # Switch to swing
                    self.knee_timers[i] = 0.0
                elif self.knee_timers[i] >= self.knee_retract_time and self.knee_states[i] == 1:
                    self.knee_states[i] = 0  # Switch back to stance
            else:  # This leg is in stance
                self.knee_states[i] = 0

    def desired_hip(self) -> float:
        """Return desired hip angle based on FSM state."""
        if self.hip_state == 0:  # Left leg swing
            return -0.3  # Swing forward
        else:  # Right leg swing
            return 0.3  # Swing forward

    def desired_knees(self) -> Tuple[float, float]:
        """Return desired knee positions based on FSM state."""
        lk_des = 0.0 if self.knee_states[0] == 0 else -0.2  # Stance: extended, Swing: retracted
        rk_des = 0.0 if self.knee_states[1] == 0 else -0.2
        return lk_des, rk_des
