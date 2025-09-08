"""
PPO-specific smoke tests.

Tests that PPO training works correctly and produces expected artifacts.
"""

import subprocess
import shlex
import pathlib


def _run(cmd):
    """Run a command and return the result."""
    return subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)


def test_cli_ppo_smoke(tmp_path):
    """Test PPO training CLI produces expected artifacts."""
    out = tmp_path / "ppo_results"
    _run(f"walker-train-ppo --config passive_walker/ppo/ppo_train.yaml --out_dir {out}")
    runs = list(out.glob("*-ppo"))
    assert runs, "PPO run dir missing"
    assert (runs[0]/"train.csv").exists()


def test_ppo_meta_json_structure(tmp_path):
    """Test that PPO produces properly structured meta.json."""
    out = tmp_path / "ppo_results"
    _run(f"walker-train-ppo --config passive_walker/ppo/ppo_train.yaml --out_dir {out}")
    runs = list(out.glob("*-ppo"))
    meta_file = runs[0] / "meta.json"
    
    import json
    with open(meta_file) as f:
        meta = json.load(f)
    
    # Check required fields
    assert meta["stage"] == "ppo_train"
    assert "git_sha" in meta
    assert "config_path" in meta
    assert "constraints" in meta
    assert "resolved_cfg" in meta
    assert "seeds" in meta
    assert "env" in meta["seeds"]
    assert "torch" in meta["seeds"]


def test_ppo_csv_structure(tmp_path):
    """Test that PPO produces properly structured train.csv."""
    out = tmp_path / "ppo_results"
    _run(f"walker-train-ppo --config passive_walker/ppo/ppo_train.yaml --out_dir {out}")
    runs = list(out.glob("*-ppo"))
    csv_file = runs[0] / "train.csv"
    
    import csv
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Check CSV structure
    assert len(rows) > 0, "CSV should have data rows"
    assert "update" in rows[0]
    assert "ep_len_mean" in rows[0]
    assert "ep_ret_mean" in rows[0]
    assert "loss_pi" in rows[0]
    assert "loss_v" in rows[0]
    
    # Check data types
    for row in rows:
        assert int(row["update"]) >= 0
        assert float(row["ep_len_mean"]) > 0
        assert float(row["ep_ret_mean"]) >= 0
        assert float(row["loss_pi"]) >= 0
        assert float(row["loss_v"]) >= 0


def test_ppo_jax_flags_warning(tmp_path):
    """Test that PPO shows warnings for JAX flags."""
    out = tmp_path / "ppo_results"
    
    # Test with JAX flags enabled by modifying the config
    import yaml
    import tempfile
    
    # Load the config and enable JAX flags
    with open("passive_walker/ppo/ppo_train.yaml") as f:
        config = yaml.safe_load(f)
    
    config["ppo"]["use_jax_pd"] = True
    config["ppo"]["use_jax_reward"] = True
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name
    
    try:
        result = _run(f"walker-train-ppo --config {temp_config} --out_dir {out}")
        
        # Should show warnings about JAX flags
        assert "use_jax_pd=True (stub)" in result.stdout or "use_jax_reward=True (OK)" in result.stdout
    finally:
        import os
        os.unlink(temp_config)


def test_ppo_constraints_support(tmp_path):
    """Test that PPO supports constraints flag."""
    out = tmp_path / "ppo_results"
    constraint_file = "passive_walker/configs/constraints/dr_light.yaml"
    
    if pathlib.Path(constraint_file).exists():
        cmd = f"walker-train-ppo --config passive_walker/ppo/ppo_train.yaml --constraints {constraint_file} --out_dir {out}"
        _run(cmd)
        
        # Check that constraints were applied
        runs = list(out.glob("*-ppo"))
        meta_file = runs[0] / "meta.json"
        
        import json
        with open(meta_file) as f:
            meta = json.load(f)
        
        assert constraint_file in meta["constraints"]


def test_ppo_vecenv_smoke(tmp_path):
    """Test PPO with vectorized environment."""
    out = tmp_path / "ppo_results"
    
    # Create a temporary config with vectorized settings
    import yaml
    import tempfile
    
    with open("passive_walker/ppo/ppo_train.yaml") as f:
        config = yaml.safe_load(f)
    
    config["ppo"]["num_envs"] = 2
    config["ppo"]["steps_per_env"] = 16
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config = f.name
    
    try:
        cmd = f"walker-train-ppo --config {temp_config} --out_dir {out}"
        _run(cmd)
        
        # Check that vectorized run completed
        runs = list(out.glob("*-ppo"))
        assert runs, "PPO run dir missing"
        assert (runs[0]/"train.csv").exists()
        
        # Check that vectorized environment was used
        import json
        with open(runs[0]/"meta.json") as f:
            meta = json.load(f)
        
        assert meta["config"]["ppo"]["num_envs"] == 2
        assert meta["config"]["ppo"]["steps_per_env"] == 16
        
    finally:
        import os
        os.unlink(temp_config)
