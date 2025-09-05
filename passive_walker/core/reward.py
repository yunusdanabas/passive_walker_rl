"""
Reward API:
- load_preset(name) -> dict
- compute_reward(obs, act, info, params) -> float
- minimal/default/aggressive handled via params dict.
"""


def load_preset(params: dict):
    return params


def compute_reward(*, dx, pitch_abs, u_abs_sum, torso_z, extras, params) -> float:
    # TODO: implement (Step 2)
    return 0.0
