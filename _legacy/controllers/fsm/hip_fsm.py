# passive_walker/controllers/fsm/hip_fsm.py
"""
Hip FSM controller for passive walker.
Runs environment's hip-FSM and returns actuator value.
"""

from passive_walker.envs.mujoco_env import PassiveWalkerEnv


class HipFSMController:
    """Wrapper for environment's hip-FSM controller. No internal state."""

    def reset(self) -> None:
        """No-op as controller has no state."""
        pass

    def action(self, env: PassiveWalkerEnv) -> float:
        """
        Run hip FSM and return actuator value.
        
        Args:
            env: Walker environment
        Returns:
            Hip actuator control signal
        """
        env.controller_fsm_hip()
        return float(env.data.ctrl[env.hip_pos_actuator_id])
