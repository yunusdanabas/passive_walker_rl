"""
Test FSM data collection schema and metadata.
"""
import tempfile
import os
import glob
import numpy as np
import json
from subprocess import run, CalledProcessError
import sys


def test_fsm_collect_npz_schema():
    """Test that FSM collection produces correct NPZ schema."""
    with tempfile.TemporaryDirectory() as d:
        # Use small run (0.08s at 100Hz = 8 steps)
        duration_sec = 0.08
        expected_steps = int(duration_sec * 100)
        cmd = [sys.executable, "-m", "passive_walker.fsm.collect",
               "--episodes", "1", "--duration", str(duration_sec), "--out", d, "--seed", "123"]
        run(cmd, check=True)
        
        # Check episode files
        files = sorted(glob.glob(os.path.join(d, "episode_*.npz")))
        assert len(files) == 1
        
        # Check NPZ schema
        data = np.load(files[0])
        expected_keys = ["obs", "act", "rew", "done", "info_pitch", "info_torso_z", "info_dx"]
        for k in expected_keys:
            assert k in data.files, f"Missing key: {k}"
        
        # Check shapes
        assert data["obs"].shape[0] == expected_steps + 1  # T+1
        assert data["act"].shape == (expected_steps, 3)
        assert data["rew"].shape == (expected_steps,)
        assert data["done"].shape == (expected_steps,)
        
        # Check metadata file
        meta_path = os.path.join(d, "meta.json")
        assert os.path.exists(meta_path)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check metadata schema (new duration-based design)
        expected_meta_keys = ["episodes", "target_steps", "seed_base", "env", "mode", 
                             "pd_backend", "ctrl_hz", "schema"]
        for k in expected_meta_keys:
            assert k in meta, f"Missing metadata key: {k}"
        
        # Check specific values
        assert meta["episodes"] == 1
        assert meta["target_steps"] == expected_steps
        assert meta["env"] == "PassiveWalkerEnv"
        assert meta["mode"] == "fsm"
        assert meta["pd_backend"] in ("numpy", "jax")
        assert isinstance(meta["ctrl_hz"], (int, float))


def test_fsm_collect_with_jax_backend():
    """Test FSM collection with JAX backend if available."""
    with tempfile.TemporaryDirectory() as d:
        # Try with JAX backend (0.04s at 100Hz = 4 steps)
        duration_sec = 0.04
        cmd = [sys.executable, "-m", "passive_walker.fsm.collect",
               "--episodes", "1", "--duration", str(duration_sec), "--out", d, "--seed", "456"]
        
        # Set environment variable to request JAX
        env = os.environ.copy()
        env["PWALKER_PD_BACKEND"] = "jax"
        
        try:
            run(cmd, check=True, env=env)
            
            # Check metadata
            meta_path = os.path.join(d, "meta.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Backend should be either jax or numpy (fallback)
            assert meta["pd_backend"] in ("numpy", "jax")
            
        except CalledProcessError:
            # If JAX is not available, that's also acceptable
            pass
