"""
Loss functions for training the omega estimator.
Implements multiple loss variants based on the mathematical derivation in chapter3.tex.
"""

import torch
import torch.nn as nn


class LossFactory:
    """
    Factory class for creating different loss functions.
    
    Available loss types:
    - 'omega_hat': Main loss from equation 567 (standard formulation)
    - 'normalized': Normalized by sigma
    - 'relative': Relative error formulation
    - 'sigma_cal': Uses calculated sigma standard deviation
    """
    
    @staticmethod
    def get_loss(loss_type='omega_hat', image_dim=3072):
        """
        Get a loss function based on the specified type.
        
        Args:
            loss_type (str): Type of loss function
            image_dim (int): Dimensionality of the image (d = C * H * W)
        
        Returns:
            Loss function object
        
        Raises:
            ValueError: If loss_type is not recognized
        """
        if loss_type == 'omega_hat':
            return OmegaHatLoss(image_dim=image_dim)
        elif loss_type == 'normalized':
            return NormalizedLoss()
        elif loss_type == 'relative':
            return RelativeLoss()
        elif loss_type == 'sigma_cal':
            return SigmaCalLoss()
        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}. "
                f"Expected 'omega_hat', 'normalized', 'relative', or 'sigma_cal'."
            )


class OmegaHatLoss(nn.Module):
    """
    Main loss function from equation 567.
    
    L = E[(omega_hat(x, sigma) - ||x - x_tilde||^2 / sigma^3)^2]
    
    This is the theoretically derived loss for training omega_phi to estimate
    the noise-level score gradient.
    
    Args:
        image_dim (int): Dimensionality of the image (d = C * H * W = 3072 for CIFAR-10)
    """
    
    def __init__(self, image_dim=3072):
        super(OmegaHatLoss, self).__init__()
        self.image_dim = image_dim
        self.mse = nn.MSELoss()
    
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute the omega_hat loss.
        
        Args:
            output (Tensor): Model prediction of shape (batch_size, 1) or (batch_size,)
            clean_images (Tensor): Clean images x_tilde of shape (batch_size, C, H, W)
            noisy_images (Tensor): Noisy images x of shape (batch_size, C, H, W)
            sigma (Tensor): Noise levels of shape (batch_size,)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure output is flattened
        output = output.squeeze()
        
        # Compute ||x - x_tilde||^2 for each image
        diff = noisy_images - clean_images
        diff_flat = diff.view(diff.size(0), -1)
        norm_squared = (diff_flat ** 2).sum(dim=1)
        
        # Compute target: ||x - x_tilde||^2 / sigma^3
        target = norm_squared / (sigma ** 3)
        
        # Compute MSE loss
        loss = self.mse(output, target)
        
        return loss


class NormalizedLoss(nn.Module):
    """
    Normalized loss: L = E[(output - sigma)^2 / sigma]
    
    This loss normalizes the error by sigma, giving more weight to predictions
    at lower noise levels.
    """
    
    def __init__(self):
        super(NormalizedLoss, self).__init__()
    
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute the normalized loss.
        
        Args:
            output (Tensor): Model prediction of shape (batch_size, 1) or (batch_size,)
            clean_images (Tensor): Clean images (unused in this loss)
            noisy_images (Tensor): Noisy images (unused in this loss)
            sigma (Tensor): Noise levels of shape (batch_size,)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure output is flattened
        output = output.squeeze()
        
        # Compute normalized squared error
        squared_error = (output - sigma) ** 2
        normalized_loss = squared_error / sigma
        
        return normalized_loss.mean()


class RelativeLoss(nn.Module):
    """
    Relative loss: L = E[(output/sigma - 1.0)^2]
    
    This loss measures the relative error, treating all noise levels equally
    in proportion to their magnitude.
    """
    
    def __init__(self):
        super(RelativeLoss, self).__init__()
    
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute the relative loss.
        
        Args:
            output (Tensor): Model prediction of shape (batch_size, 1) or (batch_size,)
            clean_images (Tensor): Clean images (unused in this loss)
            noisy_images (Tensor): Noisy images (unused in this loss)
            sigma (Tensor): Noise levels of shape (batch_size,)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure output is flattened
        output = output.squeeze()
        
        # Compute relative error
        relative_error = output / sigma - 1.0
        loss = (relative_error ** 2).mean()
        
        return loss


class SigmaCalLoss(nn.Module):
    """
    Loss using calculated sigma standard deviation.
    
    L = E[(output - (sigma_cal_std - sigma))^2]
    
    where sigma_cal_std is the calculated standard deviation from the actual noise added.
    This is an alternative formulation from the notebook experiments.
    """
    
    def __init__(self):
        super(SigmaCalLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, output, clean_images, noisy_images, sigma):
        """
        Compute the sigma_cal loss.
        
        Args:
            output (Tensor): Model prediction of shape (batch_size, 1) or (batch_size,)
            clean_images (Tensor): Clean images x_tilde of shape (batch_size, C, H, W)
            noisy_images (Tensor): Noisy images x of shape (batch_size, C, H, W)
            sigma (Tensor): Noise levels of shape (batch_size,)
        
        Returns:
            Scalar loss tensor
        """
        # Ensure output is flattened
        output = output.squeeze()
        
        # Compute actual noise
        noise = noisy_images - clean_images
        
        # Compute standard deviation of noise per image
        noise_flat = noise.view(noise.size(0), -1)
        sigma_cal_std = torch.std(noise_flat, dim=1)
        
        # Compute target: sigma_cal_std - sigma
        target = sigma_cal_std - sigma
        
        # Compute MSE loss
        loss = self.mse(output, target)
        
        return loss


def get_available_losses():
    """
    Get a list of available loss function types.
    
    Returns:
        List of loss type strings
    """
    return ['omega_hat', 'normalized', 'relative', 'sigma_cal']

