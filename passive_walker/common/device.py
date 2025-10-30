"""
Common device utilities.

Provides minimal helpers to keep GPU usage optional and simple.
"""

from __future__ import annotations


def pick_torch_device(prefer_gpu: bool) -> str:
    """
    Pick PyTorch device string based on availability and preference.

    Args:
        prefer_gpu: Whether to prefer CUDA if available

    Returns:
        "cuda" if prefer_gpu and CUDA is available, else "cpu"
    """
    try:
        import torch  # local import to avoid hard dependency at import time
        if prefer_gpu and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


