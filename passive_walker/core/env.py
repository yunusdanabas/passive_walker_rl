"""
Unified PassiveWalkerEnv.

- Single class with two modes, toggled via config.mode: "fsm" or "research".
- Reads dataclasses from YAML.
- Uses controller.PDController (FSM logic pluggable).
- Uses reward.compute_reward(...) with chosen preset.
"""


class PassiveWalkerEnv:
    def __init__(self, cfg):
        # TODO: implement (Step 1–2)
        pass
