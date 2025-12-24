"""
Output transformation module for omega estimator models.

This module provides transformations to convert model outputs to omega_hat (ω̂)
during inference and evaluation. The transformations are applied AFTER training,
not during loss computation.

Evaluation Target Formula:
--------------------------
All models are evaluated using: ω̂_target = ||x - x̃||² / σ³
where:
  - x: clean image
  - x̃: noisy image
  - σ: noise level

This is mathematically equivalent to: ω̂_target = ||ε||² / σ
where ε = (x̃ - x) / σ

Output Transformations:
-----------------------
For loss types 1-2: No transformation (model outputs omega_hat directly)
For loss type 9: Transform chi-squared z-score to omega_hat

Author: Sigma Score Estimator Team
Date: 2025
"""

import torch
import numpy as np


class OutputTransform:
    """
    Base class for output transformations.
    
    Transforms are applied during inference to convert model outputs
    to omega_hat for use in sampling algorithms.
    """
    
    def apply(self, output, sigma=None, **kwargs):
        """
        Apply transformation to model output.
        
        Args:
            output (Tensor): Raw model output, shape (batch_size,) or (batch_size, 1)
            sigma (Tensor, optional): Noise level, shape (batch_size,)
            **kwargs: Additional arguments for specific transforms
            
        Returns:
            Tensor: Transformed output (omega_hat)
        """
        raise NotImplementedError("Subclasses must implement apply()")
    
    def __call__(self, output, sigma=None, **kwargs):
        """Convenience method to call apply()."""
        return self.apply(output, sigma=sigma, **kwargs)


class IdentityTransform(OutputTransform):
    """
    Identity transformation (no transformation).
    
    Used for loss types 1-2 where model directly outputs omega_hat.
    """
    
    def __init__(self):
        pass
    
    def apply(self, output, sigma=None, **kwargs):
        """
        Return output unchanged.
        
        Args:
            output (Tensor): Model output (already omega_hat)
            sigma (Tensor, optional): Not used
            
        Returns:
            Tensor: Same as input
        """
        return output


class ChiZScoreTransform(OutputTransform):
    """
    Transform chi-squared z-score to omega_hat.
    
    Transformation: omega_hat = (omega_acute * sqrt(2*d) + d) / sigma
    
    Used for loss type 9 (omega_chi_zscore) where model outputs
    omega_acute = (||epsilon||^2 - d) / sqrt(2*d) (z-score).
    
    This transformation requires sigma to be provided.
    """
    
    def __init__(self, image_dim=3072, epsilon=1e-8):
        """
        Args:
            image_dim (int): Dimensionality of images (d)
            epsilon (float): Small constant to prevent division by zero
        """
        self.image_dim = image_dim
        self.sqrt_2d = np.sqrt(2 * image_dim)
        self.epsilon = epsilon
    
    def apply(self, output, sigma=None, **kwargs):
        """
        Transform z-score to omega_hat.
        
        Args:
            output (Tensor): Model output (z-score)
            sigma (Tensor): Noise level (REQUIRED), shape (batch_size,)
            
        Returns:
            Tensor: omega_hat = (output * sqrt(2*d) + d) / sigma
            
        Raises:
            ValueError: If sigma is not provided
        """
        if sigma is None:
            raise ValueError("ChiZScoreTransform requires sigma to be provided")
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Transform: omega_hat = (omega_acute * sqrt(2*d) + d) / sigma
        # output is the z-score: (||epsilon||^2 - d) / sqrt(2*d)
        # Reverse to get ||epsilon||^2: ||epsilon||^2 = output * sqrt(2*d) + d
        # Then: omega_hat = ||epsilon||^2 / sigma
        epsilon_norm_sq = output * self.sqrt_2d + self.image_dim
        omega_hat = epsilon_norm_sq / (sigma + self.epsilon)
        
        return omega_hat


class TransformFactory:
    """
    Factory class to create appropriate output transform for a loss type.
    """
    
    # Mapping of loss types to transform classes
    TRANSFORM_MAP = {
        # Loss types 1-2: Direct omega_hat estimation, no transform
        'omega_hat': IdentityTransform,
        'omega_epsilon': IdentityTransform,
        
        # Loss type 9: Chi-squared z-score
        'omega_chi_zscore': ChiZScoreTransform,
    }
    
    @staticmethod
    def get_transform(loss_type, image_dim=3072):
        """
        Get the appropriate output transform for a loss type.
        
        Args:
            loss_type (str): Type of loss function used during training
            image_dim (int): Dimensionality of images (default: 3072 for CIFAR-10)
            
        Returns:
            OutputTransform: Transform instance for the specified loss type
            
        Raises:
            ValueError: If loss_type is not recognized
        """
        if loss_type not in TransformFactory.TRANSFORM_MAP:
            raise ValueError(
                f"Unknown loss type: {loss_type}. "
                f"Available: {list(TransformFactory.TRANSFORM_MAP.keys())}"
            )
        
        transform_class = TransformFactory.TRANSFORM_MAP[loss_type]
        
        # Instantiate with appropriate parameters
        if transform_class == IdentityTransform:
            return transform_class()
        elif transform_class == ChiZScoreTransform:
            return transform_class(image_dim=image_dim)
        else:
            # Fallback: try to instantiate with image_dim
            return transform_class(image_dim=image_dim)
    
    @staticmethod
    def get_available_transforms():
        """
        Get list of available transform types.
        
        Returns:
            List of loss type strings that have associated transforms
        """
        return list(TransformFactory.TRANSFORM_MAP.keys())


def get_transform_info():
    """
    Print information about available output transformations.
    """
    print("=" * 80)
    print("Available Output Transformations for Omega Estimation")
    print("=" * 80)
    
    transform_info = [
        ("Loss Types 1-2", "IdentityTransform", "output (no change)"),
        ("omega_hat", "No transform", "Model outputs ω directly"),
        ("omega_epsilon", "No transform", "Model outputs ω directly"),
        ("", "", ""),
        ("Loss Type 9", "ChiZScoreTransform", "(output * √(2d) + d) / σ"),
        ("omega_chi_zscore", "z-score → ω", "ω = ||ε||² / σ"),
    ]
    
    for name, transform, formula in transform_info:
        if name == "":
            print()
        else:
            print(f"{name:25s} | {transform:30s} | {formula}")
    
    print("=" * 80)


if __name__ == "__main__":
    get_transform_info()

