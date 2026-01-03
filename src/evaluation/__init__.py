"""
Evaluation Module for MER2LATEX
================================
Metrics, evaluation, and error analysis.
"""

from .metrics import (
    compute_metrics,
    compute_exact_match,
    compute_bleu,
    compute_edit_distance,
    compute_sympy_equivalence,
    decode_predictions,
    evaluate_model,
    ErrorAnalyzer
)

__all__ = [
    'compute_metrics',
    'compute_exact_match',
    'compute_bleu',
    'compute_edit_distance',
    'compute_sympy_equivalence',
    'decode_predictions',
    'evaluate_model',
    'ErrorAnalyzer'
]
