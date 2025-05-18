from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "bc" / "hip_knee_alternatives"
DATA_DIR.mkdir(parents=True, exist_ok=True)
XML_PATH = str(Path(__file__).resolve().parents[3] / "passiveWalker_model.xml")

def set_device(use_gpu: bool):
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu" if use_gpu else "cpu")
