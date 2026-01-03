"""
EDM (Elucidating the Design Space of Diffusion Models) Denoiser Package.

This package provides functionality to use pretrained EDM models for denoising.
Based on the official implementation from NVlabs/edm repository.

Reference:
    Paper: https://arxiv.org/abs/2206.00364
    GitHub: https://github.com/NVlabs/edm

Usage:
    from src.external.edm_denoiser import load_pretrained_edm, edm_denoise
"""

import os
import sys

# Add EDM dependencies to path
_edm_path = os.path.join(os.path.dirname(__file__), 'edm')
if _edm_path not in sys.path:
    sys.path.insert(0, _edm_path)

# Import main functions from submodules
from .model_loader import load_pretrained_edm
from .core import edm_denoise
from .score import compute_score_gradient, gradient_ascent_denoise

__all__ = [
    'load_pretrained_edm',
    'edm_denoise',
    'compute_score_gradient',
    'gradient_ascent_denoise',
]

