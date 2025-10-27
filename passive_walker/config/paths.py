"""
Central Path Configuration for Passive Walker

Unified path management for all experiments and outputs.
"""

from __future__ import annotations
from pathlib import Path


def _get_project_root() -> Path:
    """Get project root directory."""
    # This file is at passive_walker/config/paths.py
    # Project root is 2 levels up
    return Path(__file__).parent.parent.parent


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


# Ensure key directories exist
for directory in [
    DATA_DIR,
    FSM_DATA_DIR,
    MODELS_DIR,
    BC_MODELS_DIR,
    PPO_MODELS_DIR,
    RUNS_DIR,
    BC_RUNS_DIR,
    PPO_RUNS_DIR,
    ANALYSIS_DIR,
    PLOTS_DIR,
    BC_PLOTS_DIR,
    PPO_PLOTS_DIR,
    REPORTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    ensure_dir_exists(directory)

