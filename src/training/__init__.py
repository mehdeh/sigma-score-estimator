"""
Training package for sigma-score-estimator.
Contains trainer, evaluator, and loss functions.
"""

from .trainer import OmegaTrainer
from .evaluator import OmegaEvaluator
from .loss_functions import LossFactory, get_available_losses

__all__ = [
    'OmegaTrainer',
    'OmegaEvaluator',
    'LossFactory',
    'get_available_losses',
]

