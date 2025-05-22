# passive_walker/controllers/nn/__init__.py
"""Neural network controllers for passive walker joints."""

from .hip_nn import HipController
from .knee_nn import KneeController
from .hip_knee_nn import HipKneeController

__all__ = ['HipController', 'KneeController', 'HipKneeController']
