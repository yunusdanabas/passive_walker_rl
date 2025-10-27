"""BC analysis module."""

from passive_walker.bc.analysis.visualize import (
    plot_training_curves,
    plot_episode_comparison,
    plot_reward_analysis
)
from passive_walker.bc.analysis.report import generate_report
from passive_walker.bc.analysis.uncertainty import (
    MCDropoutModel,
    compute_epistemic_uncertainty,
    compute_aleatoric_uncertainty
)

__all__ = [
    "plot_training_curves",
    "plot_episode_comparison",
    "plot_reward_analysis",
    "generate_report",
    "MCDropoutModel",
    "compute_epistemic_uncertainty",
    "compute_aleatoric_uncertainty",
]

