"""BC evaluation module."""

from passive_walker.bc.evaluation.evaluate import (
    evaluate_model,
    evaluate_model_comprehensive,
    ComprehensiveEvaluator,
    EvaluationResults
)
from passive_walker.bc.evaluation.play import play_policy
from passive_walker.bc.evaluation.compare_architectures import compare_architectures

__all__ = [
    "evaluate_model",
    "evaluate_model_comprehensive",
    "ComprehensiveEvaluator",
    "EvaluationResults",
    "play_policy",
    "compare_architectures",
]

