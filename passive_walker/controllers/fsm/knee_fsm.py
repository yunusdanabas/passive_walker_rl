# passive_walker/controllers/fsm/knee_fsm.py
# FSM Controller for the knee joints

from passive_walker.envs.mujoco_env import PassiveWalkerEnv
from typing import Tuple

"""
Knee FSM controller for passive walker.
Manages states and control signals for both knee joints.
"""

class KneeFSMController:
    """
    FSM controller for knee joints.
    Maintains states and produces control signals for both knees.
    """

    def __init__(self) -> None:
        """Initialize with undefined states."""
        self._state1 = None
        self._state2 = None

    def reset(self) -> None:
        """Reset both knees to stance position."""
        self._state1 = PassiveWalkerEnv.FSM_KNEE1_STANCE
        self._state2 = PassiveWalkerEnv.FSM_KNEE2_STANCE

    def step(self, env: PassiveWalkerEnv) -> Tuple[float, float]:
        """
        Run one step of knee FSM.
        
        Args:
            env: Walker environment
        Returns:
            Control signals for (left_knee, right_knee)
        """
        env.fsm_knee1 = self._state1
        env.fsm_knee2 = self._state2

        env.controller_fsm_knees()

        self._state1 = env.fsm_knee1
        self._state2 = env.fsm_knee2

        return (
            float(env.data.ctrl[env.left_knee_pos_actuator_id]),
            float(env.data.ctrl[env.right_knee_pos_actuator_id]),
        )