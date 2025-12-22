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
        'omega_chi_approx',    # Loss Type 3: Chi-squared approximation
        'omega_chi_mean',      # Loss Type 4: Expected chi-squared
        'sigma_direct',        # Loss Type 5: Direct sigma estimation
        'sigma_normalized',    # Loss Type 6: Normalized sigma estimation
        'sigma_relative',      # Loss Type 7: Relative sigma estimation
        'sigma_calibrated'     # Loss Type 8: Calibrated sigma estimation
    ]
    
    @staticmethod
    def get_loss(loss_type='omega_hat', image_dim=3072, sigma_cal=0.0):
        """
        Get loss function instance based on loss type.
        
        Args:
            loss_type: Type of loss function
            image_dim: Dimensionality of the image (e.g., 3*32*32=3072 for CIFAR-10)
            sigma_cal: Calibration parameter for sigma_calibrated loss
            
        Returns:
            Loss function instance
        """
        loss_type = loss_type.lower()
        
        if loss_type == 'omega_hat':
            return OmegaHatLoss(image_dim=image_dim)
        elif loss_type == 'omega_epsilon':
            return OmegaEpsilonLoss()
        elif loss_type == 'omega_chi_approx':
            return OmegaChiApproxLoss(image_dim=image_dim)
        elif loss_type == 'omega_chi_mean':
            return OmegaChiMeanLoss(image_dim=image_dim)
        elif loss_type == 'sigma_direct':
            return SigmaDirectLoss(image_dim=image_dim)
        elif loss_type == 'sigma_normalized':
            return SigmaNormalizedLoss(image_dim=image_dim)
        elif loss_type == 'sigma_relative':
            return SigmaRelativeLoss(image_dim=image_dim)
        elif loss_type == 'sigma_calibrated':
            return SigmaCalibratedLoss(image_dim=image_dim, sigma_cal=sigma_cal)
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
# Loss Type 3: omega_chi_approx (Chi-Squared Approximation)
# ============================================================================

class OmegaChiApproxLoss(nn.Module):
    """
    Loss Type 3: Chi-squared approximation
    
    Loss: (omega_hat - (d + sqrt(2*d) * Z) / sigma)^2
    
    Uses normal approximation: ||epsilon||^2 ≈ d + sqrt(2*d) * Z, Z ~ N(0,1)
    """
    
    def __init__(self, image_dim=3072):
        """
        Args:
            image_dim: Dimensionality of the image (d)
        """
        super(OmegaChiApproxLoss, self).__init__()
        self.image_dim = image_dim
        self.sqrt_2d = np.sqrt(2 * image_dim)
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 3.
        
        Args:
            output: Model prediction for omega_hat, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        batch_size = clean_images.size(0)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Sample Z ~ N(0, 1)
        z = torch.randn(batch_size, device=sigma.device)
        
        # Target: (d + sqrt(2*d) * Z) / sigma
        target = (self.image_dim + self.sqrt_2d * z) / sigma
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((output - target) ** 2)
        
        return loss


# ============================================================================
# Loss Type 4: omega_chi_mean (Expected Chi-Squared)
# ============================================================================

class OmegaChiMeanLoss(nn.Module):
    """
    Loss Type 4: Expected chi-squared value
    
    Loss: (omega_hat - d / sigma)^2
    
    Uses E[||epsilon||^2] = d, providing a deterministic target.
    """
    
    def __init__(self, image_dim=3072):
        """
        Args:
            image_dim: Dimensionality of the image (d)
        """
        super(OmegaChiMeanLoss, self).__init__()
        self.image_dim = image_dim
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 4.
        
        Args:
            output: Model prediction for omega_hat, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Target: d / sigma
        target = self.image_dim / sigma
        
        # Ensure output is 1D
        if output.dim() > 1:
            output = output.squeeze()
        
        # Compute MSE loss
        loss = torch.mean((output - target) ** 2)
        
        return loss


# ============================================================================
# Loss Type 5: sigma_direct (Direct Sigma Estimation)
# ============================================================================

class SigmaDirectLoss(nn.Module):
    """
    Loss Type 5: Direct sigma estimation
    
    Transformation: omega_acute = d / output
    Loss: (omega_acute - sigma)^2
    
    The network learns to predict sigma directly.
    """
    
    def __init__(self, image_dim=3072, epsilon=1e-8):
        """
        Args:
            image_dim: Dimensionality of the image (d)
            epsilon: Small constant to prevent division by zero
        """
        super(SigmaDirectLoss, self).__init__()
        self.image_dim = image_dim
        self.epsilon = epsilon
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 5.
        
        Args:
            output: Model raw output, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure output is 1D and positive
        if output.dim() > 1:
            output = output.squeeze()
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Transform: omega_acute = d / output
        # Add epsilon to prevent division by zero
        omega_acute = self.image_dim / (torch.abs(output) + self.epsilon)
        
        # Target: sigma
        target = sigma
        
        # Compute MSE loss
        loss = torch.mean((omega_acute - target) ** 2)
        
        return loss


# ============================================================================
# Loss Type 6: sigma_normalized (Normalized Sigma Estimation)
# ============================================================================

class SigmaNormalizedLoss(nn.Module):
    """
    Loss Type 6: Normalized sigma estimation
    
    Transformation: omega_acute = d / output
    Loss: (omega_acute - sigma)^2 / sigma
    
    Normalizes by sigma to weight errors at different noise levels.
    """
    
    def __init__(self, image_dim=3072, epsilon=1e-8):
        """
        Args:
            image_dim: Dimensionality of the image (d)
            epsilon: Small constant to prevent division by zero
        """
        super(SigmaNormalizedLoss, self).__init__()
        self.image_dim = image_dim
        self.epsilon = epsilon
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 6.
        
        Args:
            output: Model raw output, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure output is 1D and positive
        if output.dim() > 1:
            output = output.squeeze()
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Transform: omega_acute = d / output
        omega_acute = self.image_dim / (torch.abs(output) + self.epsilon)
        
        # Target: sigma
        target = sigma
        
        # Compute normalized loss: (omega_acute - sigma)^2 / sigma
        loss = torch.mean(((omega_acute - target) ** 2) / (sigma + self.epsilon))
        
        return loss


# ============================================================================
# Loss Type 7: sigma_relative (Relative Sigma Estimation)
# ============================================================================

class SigmaRelativeLoss(nn.Module):
    """
    Loss Type 7: Relative sigma estimation
    
    Transformation: omega_acute = d / output
    Loss: (omega_acute - sigma)^2 / sigma^2
    
    Emphasizes relative error, making the loss scale-invariant.
    """
    
    def __init__(self, image_dim=3072, epsilon=1e-8):
        """
        Args:
            image_dim: Dimensionality of the image (d)
            epsilon: Small constant to prevent division by zero
        """
        super(SigmaRelativeLoss, self).__init__()
        self.image_dim = image_dim
        self.epsilon = epsilon
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 7.
        
        Args:
            output: Model raw output, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure output is 1D and positive
        if output.dim() > 1:
            output = output.squeeze()
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Transform: omega_acute = d / output
        omega_acute = self.image_dim / (torch.abs(output) + self.epsilon)
        
        # Target: sigma
        target = sigma
        
        # Compute relative loss: (omega_acute - sigma)^2 / sigma^2
        loss = torch.mean(((omega_acute - target) ** 2) / ((sigma ** 2) + self.epsilon))
        
        return loss


# ============================================================================
# Loss Type 8: sigma_calibrated (Calibrated Sigma Estimation)
# ============================================================================

class SigmaCalibratedLoss(nn.Module):
    """
    Loss Type 8: Calibrated sigma estimation
    
    Transformation: omega_acute = d / output
    Loss: (omega_acute - (sigma_cal - sigma))^2
    
    Similar to Sigma-Cal Loss with calibration parameter.
    """
    
    def __init__(self, image_dim=3072, sigma_cal=0.0, epsilon=1e-8):
        """
        Args:
            image_dim: Dimensionality of the image (d)
            sigma_cal: Calibration parameter
            epsilon: Small constant to prevent division by zero
        """
        super(SigmaCalibratedLoss, self).__init__()
        self.image_dim = image_dim
        self.sigma_cal = sigma_cal
        self.epsilon = epsilon
        
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute Loss Type 8.
        
        Args:
            output: Model raw output, shape (batch_size, 1) or (batch_size,)
            clean_images: Clean images x_tilde, shape (batch_size, C, H, W)
            noisy_images: Noisy images x, shape (batch_size, C, H, W)
            sigma: Noise level sigma, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss value (scalar)
        """
        # Ensure output is 1D and positive
        if output.dim() > 1:
            output = output.squeeze()
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Transform: omega_acute = d / output
        omega_acute = self.image_dim / (torch.abs(output) + self.epsilon)
        
        # Target: sigma_cal - sigma
        target = self.sigma_cal - sigma
        
        # Compute MSE loss
        loss = torch.mean((omega_acute - target) ** 2)
        
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
        ("omega_chi_approx", "Chi-squared approx: (ω - (d+√(2d)Z)/σ)²"),
        ("omega_chi_mean", "Expected chi-squared: (ω - d/σ)²"),
        ("sigma_direct", "Direct sigma: (ω' - σ)², where ω'=d/output"),
        ("sigma_normalized", "Normalized sigma: (ω' - σ)²/σ"),
        ("sigma_relative", "Relative sigma: (ω' - σ)²/σ²"),
        ("sigma_calibrated", "Calibrated sigma: (ω' - (σ_cal - σ))²"),
    ]
    
    for i, (name, desc) in enumerate(loss_info, 1):
        print(f"{i}. {name:20s} : {desc}")
    
    print("=" * 80)


if __name__ == "__main__":
    print_loss_info()
