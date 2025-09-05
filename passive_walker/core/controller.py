"""
Controller API:
- PDController: computes torques given (q, qd, q_des) & gains/limits.
- FSMLogic: provides q_des for hip/knee in "fsm" mode; bypassed in "research".
"""


class PDController:
    def __init__(self, kp, kv, umin, umax):
        self.kp, self.kv, self.umin, self.umax = kp, kv, umin, umax

    def step(self, q, qd, q_des):
        # TODO: implement (Step 1)
        pass


class FSMLogic:
    def __init__(self):
        # TODO: port small, self-contained transitions (Step 1)
        pass

    def update(self, mj_data):
        pass

    def desired_positions(self):
        pass
