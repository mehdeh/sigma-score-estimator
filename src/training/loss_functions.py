"""
Loss functions for Omega estimation

This module implements various loss functions for training neural networks
to estimate omega (the squared norm of the score function) or related quantities.

Author: Sigma Score Estimator Team
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np


class LossFactory:
    """
    Factory class to create loss functions based on loss type.
    """
    
    AVAILABLE_LOSSES = [
        'omega_hat',           # Loss Type 1: Original formulation
        'omega_epsilon',       # Loss Type 2: Epsilon-based formulation
        'omega_chi_zscore',    # Loss Type 9: Chi-squared z-score
        'omega_edm',           # Computational method: EDM-based estimation (no training)
        'omega_expected',      # Computational method: Expected value estimation (no training)
        'omega_hybrid'         # Computational method: Hybrid EDM + Expected (no training)
    ]
    
    @staticmethod
    def get_loss(loss_type='omega_hat', image_dim=3072):
        """
        Get loss function instance based on loss type.
        
        Args:
            loss_type: Type of loss function
            image_dim: Dimensionality of the image (e.g., 3*32*32=3072 for CIFAR-10)
            
        Returns:
            Loss function instance, or None for computational methods that don't require loss functions
        """
        loss_type = loss_type.lower()
        
        if loss_type == 'omega_hat':
            return OmegaHatLoss(image_dim=image_dim)
        elif loss_type == 'omega_epsilon':
            return OmegaEpsilonLoss()
        elif loss_type == 'omega_chi_zscore':
            return OmegaChiZScoreLoss(image_dim=image_dim)
        elif loss_type == 'omega_edm':
            # Computational method - no loss function needed
            return None
        elif loss_type == 'omega_expected':
            # Computational method - no loss function needed
            return None
        elif loss_type == 'omega_hybrid':
            # Computational method - no loss function needed
            return None
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {LossFactory.AVAILABLE_LOSSES}")


# ============================================================================
# Loss Type 1: omega_hat (Original Formulation)
# ============================================================================

class OmegaHatLoss(nn.Module):
    """
    Loss Type 1: Original omega formulation
    
    Loss: (omega_hat - ||x - x_tilde||^2 / sigma^3)^2
    
    This is the direct estimation of omega without approximations.
    """
    
    def __init__(self, image_dim=3072):
        """
        Args:
            image_dim: Dimensionality of the image (not used in this loss but kept for consistency)
        """
        super(OmegaHatLoss, self).__init__()
        self.image_dim = image_dim
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 1.
        
        Args:
            output: Model prediction for omega_hat, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = clean_images.size(0)
        
        # Flatten images to compute L2 norm
        clean_flat = clean_images.view(batch_size, -1)
        noisy_flat = noisy_images.view(batch_size, -1)
        
        # Compute ||x - x_tilde||^2
        noise_norm_sq = torch.sum((noisy_flat - clean_flat) ** 2, dim=1)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Target: ||x - x_tilde||^2 / sigma^3
        target = noise_norm_sq / (sigma ** 3)
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((output - target) ** 2)
        
        return loss


# ============================================================================
# Loss Type 2: omega_epsilon (Epsilon-Based Formulation)
# ============================================================================

class OmegaEpsilonLoss(nn.Module):
    """
    Loss Type 2: Epsilon-based omega formulation
    
    Loss: (omega_hat - ||epsilon||^2 / sigma)^2
    
    Uses the noise epsilon directly: x = x_tilde + sigma * epsilon
    More numerically stable for small sigma.
    """
    
    def __init__(self):
        super(OmegaEpsilonLoss, self).__init__()
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 2.
        
        Args:
            output: Model prediction for omega_hat, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = clean_images.size(0)
        
        # Flatten images
        clean_flat = clean_images.view(batch_size, -1)
        noisy_flat = noisy_images.view(batch_size, -1)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Compute epsilon = (x - x_tilde) / sigma
        epsilon = (noisy_flat - clean_flat) / sigma.view(-1, 1)
        
        # Compute ||epsilon||^2
        epsilon_norm_sq = torch.sum(epsilon ** 2, dim=1)
        
        # Target: ||epsilon||^2 / sigma
        target = epsilon_norm_sq / sigma
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((output - target) ** 2)
        
        return loss


# ============================================================================
# Loss Type 9: omega_chi_zscore (Chi-Squared Z-Score)
# ============================================================================

class OmegaChiZScoreLoss(nn.Module):
    """
    Loss Type 9: Chi-squared z-score based estimation
    
    The model learns the z-score of the chi-squared distribution:
    Target: (||epsilon||^2 - d) / sqrt(2*d)
    
    This normalizes the chi-squared random variable to have approximately
    zero mean and unit variance (by the Central Limit Theorem).
    
    After training, output is transformed to omega_hat during inference:
    omega_hat = (output * sqrt(2*d) + d) / sigma
    
    Target: (||epsilon||^2 - d) / sqrt(2*d)
    Loss: (output - target)^2
    """
    
    def __init__(self, image_dim=3072):
        """
        Args:
            image_dim: Dimensionality of the image (d)
        """
        super(OmegaChiZScoreLoss, self).__init__()
        self.image_dim = image_dim
        self.sqrt_2d = np.sqrt(2 * image_dim)
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 9.
        
        Args:
            output: Model output (z-score prediction), shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = clean_images.size(0)
        
        # Flatten images
        clean_flat = clean_images.view(batch_size, -1)
        noisy_flat = noisy_images.view(batch_size, -1)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Compute epsilon = (x - x_tilde) / sigma
        epsilon = (noisy_flat - clean_flat) / sigma.view(-1, 1)
        
        # Compute ||epsilon||^2
        epsilon_norm_sq = torch.sum(epsilon ** 2, dim=1)
        
        # Target: (||epsilon||^2 - d) / sqrt(2*d)
        # This is the z-score of the chi-squared distribution
        target = (epsilon_norm_sq - self.image_dim) / self.sqrt_2d
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((output - target) ** 2)
        
        return loss


# ============================================================================
# Utility Functions
# ============================================================================

def get_available_losses():
    """
    Return list of available loss types.
    
    Returns:
        List of loss type names
    """
    return LossFactory.AVAILABLE_LOSSES.copy()


def print_loss_info():
    """
    Print information about all available loss functions.
    """
    print("=" * 80)
    print("Available Loss Functions for Omega Estimation")
    print("=" * 80)
    
    loss_info = [
        ("omega_hat", "Original formulation: (ω - ||x-x̃||²/σ³)²"),
        ("omega_epsilon", "Epsilon-based: (ω - ||ε||²/σ)²"),
        ("omega_chi_zscore", "Chi z-score: (output - (||ε||²-d)/√(2d))²"),
        ("omega_edm", "Computational: EDM-based (no training required)"),
        ("omega_expected", "Computational: Expected value d/σ (no training required)"),
        ("omega_hybrid", "Computational: Hybrid EDM+Expected (no training required)"),
    ]
    
    for i, (name, desc) in enumerate(loss_info, 1):
        print(f"{i}. {name:20s} : {desc}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_loss_info()
