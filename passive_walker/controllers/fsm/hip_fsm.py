# passive_walker/controllers/fsm/hip_fsm.py
# FSM Controller for the hip joint

from passive_walker.envs.mujoco_env import PassiveWalkerEnv

class HipFSMController:
    """
    Wraps the env.controller_fsm_hip() logic so we can
    1) compute exactly the hip action the FSM would have taken, and
    2) feed that back as an 'external_action' for logging.
    """

    def __init__(self):
        self._state = None

    def reset(self):
        # Reset state to env’s default
        self._state = PassiveWalkerEnv.FSM_HIP_LEG2_SWING

    def step(self, env: PassiveWalkerEnv) -> float:
        # Copy env’s FSM state into the controller, so it stays in sync
        env.fsm_hip = self._state

        # Run the FSM logic (this writes into env.data.ctrl[hip])
        env.controller_fsm_hip()

        # Read back the new FSM state & output
        self._state = env.fsm_hip
        return float(env.data.ctrl[env.hip_pos_actuator_id])