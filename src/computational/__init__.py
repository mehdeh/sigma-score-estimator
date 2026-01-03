"""
Computational methods for omega estimation.

This module contains non-trainable computational methods for estimating omega_hat
using pretrained models or closed-form computations.

Unlike the models in src.models which require training, these methods provide
direct computational estimates of omega_hat without any training phase.

Available Methods:
-----------------
- EDMOmegaEstimator: Uses pretrained EDM denoiser to compute ||x - x̃||² / σ³
- ExpectedOmegaEstimator: Uses statistical expectation to compute d / σ
"""

from .edm_estimator import EDMOmegaEstimator
from .expected_estimator import ExpectedOmegaEstimator

__all__ = [
    'EDMOmegaEstimator',
    'ExpectedOmegaEstimator',
]

