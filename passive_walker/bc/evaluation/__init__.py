"""BC evaluation module.

``play`` runs closed-loop playback of trained BC checkpoints. ``evaluate``
(the comprehensive evaluator) is imported explicitly by its users rather than
here, so that playback does not depend on it.
"""

from passive_walker.bc.evaluation.play import play_jax, play_torch

__all__ = [
    "play_torch",
    "play_jax",
]
