"""
Data module for MER2LATEX
==========================
Contains dataset loading and evaluation utilities.
"""

from src.data.dataset import MERDataset, get_dataloader
from src.data.evaluation import (
    compute_metrics,
    decode_predictions,
    evaluate_model,
    ErrorAnalyzer
)

__all__ = [
    'MERDataset',
    'get_dataloader',
    'compute_metrics',
    'decode_predictions',
    'evaluate_model',
    'ErrorAnalyzer'
]
