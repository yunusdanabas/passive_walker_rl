# passive_walker/bc/__init__.py
from importlib import import_module

# auto-import hip_mse so `python -m passive_walker.bc.hip_mse.full_pipeline` works
import_module("passive_walker.bc.hip_mse")
