"""
Tests for CLI dry-run functionality.
"""

import subprocess
import sys


def test_fsm_dry_run():
    """Test FSM collection dry-run."""
    cmd = [sys.executable, "-m", "passive_walker.fsm.collect", "--config", "passive_walker/fsm/fsm_collect.yaml", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout


def test_bc_dry_run():
    """Test BC training dry-run."""
    cmd = [sys.executable, "-m", "passive_walker.bc.train", "--config", "passive_walker/bc/bc_train.yaml", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout


def test_ppo_dry_run():
    """Test PPO training dry-run."""
    cmd = [sys.executable, "-m", "passive_walker.ppo.train", "--config", "passive_walker/ppo/ppo_train.yaml", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout


def test_fsm_print_config():
    """Test FSM collection print-config."""
    cmd = [sys.executable, "-m", "passive_walker.fsm.collect", "--config", "passive_walker/fsm/fsm_collect.yaml", "--print-config"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout


def test_bc_print_config():
    """Test BC training print-config."""
    cmd = [sys.executable, "-m", "passive_walker.bc.train", "--config", "passive_walker/bc/bc_train.yaml", "--print-config"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout


def test_ppo_print_config():
    """Test PPO training print-config."""
    cmd = [sys.executable, "-m", "passive_walker.ppo.train", "--config", "passive_walker/ppo/ppo_train.yaml", "--print-config"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "bc" in result.stdout and "control" in result.stdout
