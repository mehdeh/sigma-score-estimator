"""
Training package for sigma-score-estimator.
Contains trainer, evaluator, loss functions, and output transforms.
"""

from .trainer import OmegaTrainer
from .evaluator import OmegaEvaluator
from .loss_functions import LossFactory, get_available_losses
from .output_transforms import (
    OutputTransform,
    IdentityTransform,
    SigmaToOmegaTransform,
    SigmaCalibratedTransform,
    ChiZScoreTransform,
    TransformFactory,
)

__all__ = [
    'OmegaTrainer',
    'OmegaEvaluator',
    'LossFactory',
    'get_available_losses',
    'OutputTransform',
    'IdentityTransform',
    'SigmaToOmegaTransform',
    'SigmaCalibratedTransform',
    'ChiZScoreTransform',
    'TransformFactory',
]

