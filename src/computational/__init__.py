"""
Computational methods for omega estimation.

This module contains non-trainable computational methods for estimating omega_hat
using pretrained models or closed-form computations.

Unlike the models in src.models which require training, these methods provide
direct computational estimates of omega_hat without any training phase.
"""

from .edm_estimator import EDMOmegaEstimator

__all__ = [
    'EDMOmegaEstimator',
]

