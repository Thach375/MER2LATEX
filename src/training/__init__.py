"""
Training module for MER2LATEX
==============================
Contains training utilities and Trainer class.
"""

from src.training.trainer import Trainer, train_model
from src.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TrainingLogger
)

__all__ = [
    'Trainer',
    'train_model',
    'EarlyStopping',
    'ModelCheckpoint',
    'TrainingLogger'
]
