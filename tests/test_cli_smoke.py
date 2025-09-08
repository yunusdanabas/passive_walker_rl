"""
CLI smoke tests for all stage commands.

Tests that all CLI commands run without crashing and produce expected artifacts.
"""

import subprocess
import shlex
import pathlib


def _run(cmd):
    """Run a command and return the result."""
    return subprocess.run(shlex.split(cmd), check=True, capture_output=True, text=True)


def test_cli_demo_fsm_20s():
    """Test FSM demo runs for 20 seconds headless."""
    # use smaller seconds if you must—but spec asked 20s
    _run("walker-demo --no-gui --seconds 20")


def test_cli_collect_fsm(tmp_path):
    """Test FSM collection CLI produces expected artifacts."""
    out = tmp_path / "raw"
    res = tmp_path / "res"
    cmd = f"walker-collect-fsm --config passive_walker/fsm/fsm_collect.yaml --num_episodes 1 --rollout_len 40 --seed 11 --output_dir {out} --results_dir {res}"
    _run(cmd)
    assert any(p.suffix==".npz" for p in out.glob("*.npz"))
    assert (res).exists()


def test_cli_bc_overfit(tmp_path):
    """Test BC training with overfit_tiny flag."""
    # Prepare tiny dataset
    out = tmp_path / "raw"; res = tmp_path / "res"
    _run(f"walker-collect-fsm --config passive_walker/fsm/fsm_collect.yaml --num_episodes 1 --rollout_len 40 --seed 13 --output_dir {out} --results_dir {res}")
    # Train
    _run(f"walker-train-bc --config passive_walker/bc/bc_train.yaml --data_dir {out} --overfit_tiny --seed 7 --out_dir {tmp_path/'bc_results'}")
    runs = list((tmp_path/"bc_results").glob("*-bc"))
    assert runs, "BC run dir missing"
    csv = runs[0]/"train.csv"
    assert csv.exists()


def test_cli_ppo_smoke(tmp_path):
    """Test PPO training CLI produces expected artifacts."""
    out = tmp_path / "ppo_results"
    _run(f"walker-train-ppo --config passive_walker/ppo/ppo_train.yaml --out_dir {out}")
    runs = list(out.glob("*-ppo"))
    assert runs, "PPO run dir missing"
    assert (runs[0]/"train.csv").exists()


def test_cli_help_messages():
    """Test that all CLI commands show help messages."""
    commands = [
        "walker-demo --help",
        "walker-collect-fsm --help", 
        "walker-train-bc --help",
        "walker-train-ppo --help"
    ]
    
    for cmd in commands:
        result = _run(cmd)
        assert "usage:" in result.stdout.lower() or "help" in result.stdout.lower()


def test_cli_constraints_flag():
    """Test that constraints flag works with CLI commands."""
    # Test with a constraint file
    constraint_file = "passive_walker/configs/constraints/dr_light.yaml"
    if pathlib.Path(constraint_file).exists():
        cmd = f"walker-collect-fsm --config passive_walker/fsm/fsm_collect.yaml --constraints {constraint_file} --num_episodes 1 --rollout_len 20 --seed 42"
        _run(cmd)


def test_cli_seed_determinism(tmp_path):
    """Test that CLI commands are deterministic with same seed."""
    out1 = tmp_path / "raw1"; res1 = tmp_path / "res1"
    out2 = tmp_path / "raw2"; res2 = tmp_path / "res2"
    
    # Run same command twice with same seed
    cmd1 = f"walker-collect-fsm --config passive_walker/fsm/fsm_collect.yaml --num_episodes 1 --rollout_len 20 --seed 42 --output_dir {out1} --results_dir {res1}"
    cmd2 = f"walker-collect-fsm --config passive_walker/fsm/fsm_collect.yaml --num_episodes 1 --rollout_len 20 --seed 42 --output_dir {out2} --results_dir {res2}"
    
    _run(cmd1)
    _run(cmd2)
    
    # Check that episodes are identical
    ep1 = list(out1.glob("*.npz"))[0]
    ep2 = list(out2.glob("*.npz"))[0]
    
    import numpy as np
    data1 = np.load(ep1)
    data2 = np.load(ep2)
    
    # Episodes should be identical with same seed
    assert np.allclose(data1['obs'], data2['obs'])
    assert np.allclose(data1['act'], data2['act'])
    assert np.allclose(data1['rew'], data2['rew'])

