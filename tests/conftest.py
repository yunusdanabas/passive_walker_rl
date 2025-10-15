import os
import numpy as np
import pytest

# Keep headless predictable. We don't render in tests.
os.environ.setdefault("MUJOCO_GL", "egl")

@pytest.fixture(autouse=True)
def _np_print_options():
    old = np.get_printoptions()
    np.set_printoptions(suppress=True, linewidth=120)
    try:
        yield
    finally:
        np.set_printoptions(**old)
