# passive_walker/utils/io.py
"""Tiny helpers for pickling Equinox/JAX models or NumPy arrays."""

from pathlib import Path
import pickle

def save_pickle(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[utils.io] saved → {path}")

def load_pickle(path: str | Path):
    path = Path(path)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[utils.io] loaded ← {path}")
    return obj
