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
    ChiZScoreTransform,
    TransformFactory,
)
from .evaluation_utils import (
    compute_omega_hat_target,
    compute_raw_target,
    compute_evaluation_metrics,
    log_evaluation_metrics,
)

__all__ = [
    'OmegaTrainer',
    'OmegaEvaluator',
    'LossFactory',
    'get_available_losses',
    'OutputTransform',
    'IdentityTransform',
    'ChiZScoreTransform',
    'TransformFactory',
    'compute_omega_hat_target',
    'compute_raw_target',
    'compute_evaluation_metrics',
    'log_evaluation_metrics',
]

