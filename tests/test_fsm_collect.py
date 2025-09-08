import json
import os
import sys
import glob
import shutil
import numpy as np
import pytest
from pathlib import Path
import subprocess

def _try_import_collect():
    try:
        from passive_walker.fsm import collect as mod
        return getattr(mod, "collect", None)
    except Exception:
        return None

def _validate_schema(npz_path: str, expected_T: int):
    d = np.load(npz_path)
    assert d["obs"].ndim == 2 and d["obs"].shape[0] in (expected_T, expected_T+1)
    assert d["act"].shape == (expected_T, 3)
    assert d["rew"].shape == (expected_T,)
    assert d["done"].shape == (expected_T,)
    # optional info keys
    for k in ("info_pitch", "info_torso_z", "info_dx"):
        assert k in d.files

def test_collect_minimal(tmp_path: Path):
    outdir = tmp_path / "fsm_min"
    outdir.mkdir(parents=True, exist_ok=True)

    collect_fn = _try_import_collect()
    if collect_fn is not None:
        collect_fn(episodes=1, steps=16, outdir=str(outdir), seed=123)
    else:
        # Fallback: run as a module
        cmd = [sys.executable, "-m", "passive_walker.fsm.collect",
               "--episodes", "1", "--steps", "16", "--out", str(outdir), "--seed", "123"]
        subprocess.run(cmd, check=True)

    files = sorted(glob.glob(str(outdir / "episode_*.npz")))
    assert len(files) == 1
    _validate_schema(files[0], expected_T=16)

    # meta.json is helpful but optional; if present, sanity check
    meta = outdir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        assert "episodes" in m and "steps_per_episode" in m and "mode" in m

@pytest.mark.slow
def test_collect_determinism(tmp_path: Path):
    # Two independent outputs with same seed must match
    A = tmp_path / "A"; B = tmp_path / "B"
    A.mkdir(); B.mkdir()
    cmdA = [sys.executable, "-m", "passive_walker.fsm.collect",
            "--episodes", "1", "--steps", "8", "--out", str(A), "--seed", "777"]
    cmdB = [sys.executable, "-m", "passive_walker.fsm.collect",
            "--episodes", "1", "--steps", "8", "--out", str(B), "--seed", "777"]
    subprocess.run(cmdA, check=True); subprocess.run(cmdB, check=True)

    a = np.load(sorted(glob.glob(str(A / "episode_*.npz")))[0])
    b = np.load(sorted(glob.glob(str(B / "episode_*.npz")))[0])

    for key in ("obs", "act", "rew", "done", "info_pitch", "info_torso_z", "info_dx"):
        assert np.allclose(a[key], b[key]), f"Mismatch in {key}"
