import numpy as np
import pytest

from passive_walker.core.env import PassiveWalkerEnv

def test_reset_shapes_and_types():
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (11,)
    env.close()

def test_step_contract_and_info_keys():
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    obs, _ = env.reset(seed=123)
    a = np.zeros(3, dtype=np.float32)
    obs2, r, done, info = env.step(a)
    # observation type/shape
    assert isinstance(obs2, np.ndarray) and obs2.shape == (11,)
    # reward scalar
    assert isinstance(r, float)
    # done bool
    assert isinstance(done, (bool, np.bool_))
    # minimal info keys present
    for k in ("time", "dx", "pitch_abs", "torso_z"):
        assert k in info
    env.close()

def rollout_obs(env, T=20):
    a = np.zeros(3, dtype=np.float32)
    out = [env.reset(seed=123)[0]]
    for _ in range(T):
        o, r, d, info = env.step(a)
        out.append(o.copy())
    return np.stack(out)

def test_determinism_zero_actions():
    env1 = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env2 = PassiveWalkerEnv(mode="fsm", use_gui=False)

    obs1 = rollout_obs(env1, T=25)
    obs2 = rollout_obs(env2, T=25)

    assert np.allclose(obs1, obs2), "Two runs with same seed/action should match exactly"
    env1.close(); env2.close()

@pytest.mark.slow
def test_env_longer_roll_no_crash():
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env.reset(seed=2024)
    a = np.zeros(3, dtype=np.float32)
    for _ in range(200):
        _, _, done, _ = env.step(a)
        if done:
            break
    env.close()
