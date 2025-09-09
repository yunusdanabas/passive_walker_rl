"""
Test FSM collector round-trip functionality.
"""
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
from passive_walker.fsm.collect import collect


def test_collector_roundtrip():
    """Test collector produces correct NPZ schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "test_collect"
        outdir.mkdir()
        
        # Collect 1 episode with 256 steps
        collect(
            episodes=1,
            steps=256,
            outdir=str(outdir),
            seed=123
        )
        
        # Check files were created
        npz_file = outdir / "episode_000000.npz"
        meta_file = outdir / "meta.json"
        
        assert npz_file.exists(), "NPZ file not created"
        assert meta_file.exists(), "Meta file not created"
        
        # Load and check NPZ schema
        data = np.load(npz_file)
        
        # Check required arrays exist
        required_arrays = ["obs", "act", "rew", "done"]
        for key in required_arrays:
            assert key in data.files, f"Missing array: {key}"
        
        # Check shapes
        obs = data["obs"]
        act = data["act"]
        rew = data["rew"]
        done = data["done"]
        
        # obs should be (T+1, 11) - includes initial state
        assert obs.shape == (257, 11), f"obs shape incorrect: {obs.shape}"
        
        # act, rew, done should be (T,)
        assert act.shape == (256, 3), f"act shape incorrect: {act.shape}"
        assert rew.shape == (256,), f"rew shape incorrect: {rew.shape}"
        assert done.shape == (256,), f"done shape incorrect: {done.shape}"
        
        # Check data types
        assert obs.dtype == np.float32, f"obs dtype incorrect: {obs.dtype}"
        assert act.dtype == np.float32, f"act dtype incorrect: {act.dtype}"
        assert rew.dtype == np.float32, f"rew dtype incorrect: {rew.dtype}"
        assert done.dtype == bool, f"done dtype incorrect: {done.dtype}"
        
        # Check values are finite
        assert np.all(np.isfinite(obs)), "obs contains non-finite values"
        assert np.all(np.isfinite(act)), "act contains non-finite values"
        assert np.all(np.isfinite(rew)), "rew contains non-finite values"
        
        # Check done flags are reasonable (mostly False, maybe True at end)
        assert np.sum(done) <= 1, f"Too many done=True: {np.sum(done)}"
        
        # Check optional info arrays if they exist
        info_keys = ["info_pitch", "info_torso_z", "info_dx"]
        for key in info_keys:
            if key in data.files:
                info_array = data[key]
                assert info_array.shape == (256,), f"{key} shape incorrect: {info_array.shape}"
                assert np.all(np.isfinite(info_array)), f"{key} contains non-finite values"
