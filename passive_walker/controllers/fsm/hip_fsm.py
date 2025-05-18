# passive_walker/controllers/fsm/hip_fsm.py
"""
Thin wrapper that runs the environment’s internal hip-FSM and
returns the hip actuator value so callers can log it.
"""

class HipFSMController:
    def reset(self):                     # no state to keep
        pass

    def action(self, env) -> float:
        """
        Runs env.controller_fsm_hip(), then reads out the hip command that
        the FSM wrote into env.data.ctrl[hip_actuator].
        """
        env.controller_fsm_hip()
        return float(env.data.ctrl[env.hip_pos_actuator_id])
