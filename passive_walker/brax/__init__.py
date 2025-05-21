# passive_walker/brax/__init__.py
# ---------------------------------------------------------------------
"""
Brax integration for the Passive Walker project
===============================================

This sub-package contains:

* `convert_xml.py` – MuJoCo XML ➜ Brax System converter
* `system.pkl.gz` – serialized **Brax System** (auto-generated)
* `walker_brax_env.py` – `brax.envs.base.Env` implementation
* training / benchmarking helpers (PPO, vectorised roll-outs, …)

Directory layout created on first import
----------------------------------------

project_root/
├─ data/
│  └─ brax/                 ← DATA_BRAX
│     └─ system.pkl.gz      ← SYSTEM_PICKLE  (cached output)
└─ passiveWalker_model.xml  ← XML_PATH  (source geometry)

The paths are exported as module attributes so every script / notebook
can just do::

    from passive_walker.brax import XML_PATH, SYSTEM_PICKLE, set_device
"""

from __future__ import annotations
from pathlib import Path

from passive_walker.constants import ROOT, XML_PATH, DATA_DIR, set_device

__all__ = [
    "__version__",
    "ROOT",
    "XML_PATH",
    "DATA_DIR",
    "DATA_BRAX",
    "SYSTEM_PICKLE",
    "RESULTS_BRAX",
    "set_device",
]

__version__ = "0.1.0"

# ----------------------------------------------------------------------
# 🔗  Brax-specific paths
# ----------------------------------------------------------------------
# All Brax-specific artefacts (pickled system, PPO checkpoints, …)
DATA_BRAX: Path = DATA_DIR / "brax"
DATA_BRAX.mkdir(parents=True, exist_ok=True)

# Cached Brax System pickle – produced by convert_xml.py
SYSTEM_PICKLE: Path = DATA_BRAX / "system.pkl.gz"

# Training / evaluation results (plots, CSVs, logs …)
RESULTS_BRAX: Path = ROOT / "results" / "brax"
RESULTS_BRAX.mkdir(parents=True, exist_ok=True)


