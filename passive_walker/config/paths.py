"""
Central Path Configuration for Passive Walker

Unified path management for all experiments and outputs. Importing this module
has no side effects: writers create the directories they need via
``ensure_dir_exists``.

The experiments root is ``$PASSIVE_WALKER_HOME/experiments`` if the environment
variable is set, otherwise ``<repo>/experiments`` for a source checkout, and
``./experiments`` (current working directory) for an installed package.
"""

from __future__ import annotations
import os
from pathlib import Path


def _get_project_root() -> Path:
    """Directory that holds ``experiments/``."""
    override = os.environ.get("PASSIVE_WALKER_HOME")
    if override:
        return Path(override).expanduser().resolve()
    # This file is at passive_walker/config/paths.py; the repo root is 2 levels up.
    source_root = Path(__file__).resolve().parent.parent.parent
    if (source_root / "pyproject.toml").is_file():
        return source_root
    return Path.cwd()


# Project root
PROJECT_ROOT = _get_project_root()

# Experiments root
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# Data paths
DATA_DIR = EXPERIMENTS_ROOT / "data"
FSM_DATA_DIR = DATA_DIR / "fsm_runs"

# Model paths
MODELS_DIR = EXPERIMENTS_ROOT / "models"
BC_MODELS_DIR = MODELS_DIR / "bc"
PPO_MODELS_DIR = MODELS_DIR / "ppo"

# Training log paths (TensorBoard, etc.)
RUNS_DIR = EXPERIMENTS_ROOT / "runs"
BC_RUNS_DIR = RUNS_DIR / "bc"
PPO_RUNS_DIR = RUNS_DIR / "ppo"

# Analysis output paths (plots, metrics, reports)
ANALYSIS_DIR = EXPERIMENTS_ROOT / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
BC_PLOTS_DIR = PLOTS_DIR / "bc"
PPO_PLOTS_DIR = PLOTS_DIR / "ppo"
REPORTS_DIR = ANALYSIS_DIR / "reports"
METRICS_DIR = ANALYSIS_DIR / "metrics"
FIGURES_DIR = ANALYSIS_DIR / "figures"


def ensure_dir_exists(path: Path | str) -> Path:
    """
    Ensure directory exists, creating parent directories if needed.

    Args:
        path: Path to ensure exists

    Returns:
        Path object (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
