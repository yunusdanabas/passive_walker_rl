"""
hip_knee_mse - BC for hip + knee controller with MSE loss.

Usage:
  python -m passive_walker.bc.hip_knee_mse.collect  [--steps N] [--gpu]
  python -m passive_walker.bc.hip_knee_mse.train    [--epochs E] [--gpu]
  python -m passive_walker.bc.hip_knee_mse.run_pipeline [--gpu]
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "bc"
DATA_DIR.mkdir(parents=True, exist_ok=True)

XML_PATH = str(Path(__file__).resolve().parents[3] / "passiveWalker_model.xml")

def set_device(use_gpu: bool):
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu" if use_gpu else "cpu")