"""
Golden FSM trajectory regression test.

Guards env/controller refactors: a nominal 5 s FSM rollout must reproduce the
stored body/joint trajectory and FSM state sequence. Tiny numerical changes
(float32 vs float64 PD, 1e-9 state perturbations) move positions by < 1e-6
over 5 s, so a 1e-4 tolerance only trips on real behavior changes.

Regenerate (only when a behavior change is intended and reviewed):
    python tests/test_fsm_golden.py
"""
from pathlib import Path

import numpy as np

from passive_walker.core.env import PassiveWalkerEnv

GOLDEN_PATH = Path(__file__).parent / "data" / "fsm_golden.npz"
N_STEPS = 500  # 5 s at 100 Hz
SEED = 0
POS_ATOL = 1e-4


def fsm_rollout(n_steps: int = N_STEPS, seed: int = SEED) -> dict:
    """Run the nominal FSM and record state read directly from MuJoCo data."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    env.reset(seed=seed)
    env.simend = 1e9
    zero = np.zeros(3, dtype=np.float32)
    qpos_idx = [env.qpos_x, env.qpos_z, env.qpos_pitch, env.qpos_hip, env.qpos_lk, env.qpos_rk]
    qpos, qdes, fsm_states = [], [], []
    for _ in range(n_steps):
        env.step(zero)
        qpos.append(env.data.qpos[qpos_idx].copy())
        qdes.append(env._qdes.copy())
        fsm_states.append([env.fsm.fsm_hip, env.fsm.fsm_knee1, env.fsm.fsm_knee2])
    env.close()
    return {
        "qpos": np.asarray(qpos, dtype=np.float64),  # x, z, pitch, hip, lk, rk
        "qdes": np.asarray(qdes, dtype=np.float64),
        "fsm_states": np.asarray(fsm_states, dtype=np.int8),
    }


def test_fsm_matches_golden_trajectory():
    golden = np.load(GOLDEN_PATH)
    out = fsm_rollout()
    np.testing.assert_array_equal(out["fsm_states"], golden["fsm_states"])
    np.testing.assert_allclose(out["qdes"], golden["qdes"], atol=1e-6)
    np.testing.assert_allclose(out["qpos"], golden["qpos"], atol=POS_ATOL)


if __name__ == "__main__":
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = fsm_rollout()
    np.savez_compressed(GOLDEN_PATH, **data)
    print(f"Wrote {GOLDEN_PATH}: final x = {data['qpos'][-1, 0]:.4f} m")
