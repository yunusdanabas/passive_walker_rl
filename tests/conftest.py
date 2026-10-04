import numpy as np
import pytest

# Tests never render. Leave MUJOCO_GL unset: forcing "egl" makes `import mujoco`
# fail on machines without EGL, while the default (glfw) is only loaded lazily.

@pytest.fixture(autouse=True)
def _np_print_options():
    old = np.get_printoptions()
    np.set_printoptions(suppress=True, linewidth=120)
    try:
        yield
    finally:
        np.set_printoptions(**old)
