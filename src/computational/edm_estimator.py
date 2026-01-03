"""
EDM-based Omega Estimator.

This module provides a computational method for estimating omega_hat using
a pretrained EDM denoiser, without requiring any training.

The estimator computes: ω̂ = ||x - x̃||² / σ³
where x̃ is the denoised image obtained from the pretrained EDM model.

Author: Sigma Score Estimator Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Union, Optional

from ..external.edm_denoiser import load_pretrained_edm, edm_denoise


class EDMOmegaEstimator(nn.Module):
    """
    Computational estimator for omega_hat using pretrained EDM denoiser.
    
    This estimator provides a training-free method to compute omega_hat by:
    1. Denoising the noisy image x using a pretrained EDM model to get x̃
    2. Computing: ω̂ = ||x - x̃||² / σ³
    
    This is a direct computational estimate of the target used in omega_hat training,
    but obtained without training a separate model.
    
    Parameters:
    -----------
    model_name : str
        Name of the pretrained EDM model to use (default: 'cifar10-uncond-ve')
        Available models:
            - 'cifar10-uncond-vp': Unconditional CIFAR-10, VP parameterization
            - 'cifar10-uncond-ve': Unconditional CIFAR-10, VE parameterization
            - 'cifar10-cond-vp': Conditional CIFAR-10, VP parameterization
            - 'cifar10-cond-ve': Conditional CIFAR-10, VE parameterization
            - 'cifar10-uncond': Alias for 'cifar10-uncond-ve'
            - 'cifar10-cond': Alias for 'cifar10-cond-ve'
    device : str or torch.device
        Device to run computations on (default: 'cuda' if available, else 'cpu')
        
    Attributes:
    -----------
    edm_model : torch.nn.Module
        The pretrained EDM denoising model
    device : torch.device
        Device where computations are performed
    config : dict
        EDM model configuration
        
    Examples:
    ---------
    >>> import torch
    >>> from src.computational import EDMOmegaEstimator
    >>> 
    >>> # Initialize estimator
    >>> estimator = EDMOmegaEstimator('cifar10-uncond-ve', device='cuda')
    >>> 
    >>> # Generate noisy images
    >>> clean_imgs = torch.randn(4, 3, 32, 32).cuda()
    >>> sigma = torch.tensor([2.0, 2.0, 2.0, 2.0]).cuda()
    >>> noise = torch.randn_like(clean_imgs) * sigma.view(-1, 1, 1, 1)
    >>> noisy_imgs = clean_imgs + noise
    >>> 
    >>> # Estimate omega_hat
    >>> with torch.no_grad():
    >>>     omega_hat = estimator(noisy_imgs, sigma)
    >>> 
    >>> print(f"Omega hat shape: {omega_hat.shape}")  # torch.Size([4])
    """
    
    def __init__(
        self,
        model_name: str = 'cifar10-uncond-ve',
        device: Optional[Union[str, torch.device]] = None
    ):
        super(EDMOmegaEstimator, self).__init__()
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Load pretrained EDM model
        print(f"Loading pretrained EDM model: {model_name}")
        self.edm_model, self.config = load_pretrained_edm(
            model_name=model_name,
            device=self.device
        )
        
        # Ensure model is in eval mode
        self.edm_model.eval()
        
        # Freeze EDM model parameters (no training needed)
        for param in self.edm_model.parameters():
            param.requires_grad = False
            
        print(f"EDM Omega Estimator initialized on {self.device}")
        print(f"Model config: {self.config}")
    
    def forward(
        self,
        noisy_images: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute omega_hat using EDM denoiser.
        
        This method:
        1. Denoises noisy_images using the EDM model to get x̃
        2. Computes ω̂ = ||x - x̃||² / σ³
        
        Parameters:
        -----------
        noisy_images : torch.Tensor
            Noisy input images of shape (batch_size, C, H, W)
        sigma : torch.Tensor
            Noise levels of shape (batch_size,) or (batch_size, 1)
        class_labels : torch.Tensor, optional
            Class labels for conditional models (shape: batch_size,)
            For unconditional models, set to None (default)
            
        Returns:
        --------
        omega_hat : torch.Tensor
            Estimated omega_hat values of shape (batch_size,)
            
        Notes:
        ------
        - All computations are done in no_grad mode since this is inference only
        - The formula is: ω̂ = ||x - x̃||² / σ³
        - This is mathematically equivalent to the target used in omega_hat training
        """
        batch_size = noisy_images.size(0)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Denoise using EDM model (no gradient needed)
        with torch.no_grad():
            denoised_images = edm_denoise(
                model=self.edm_model,
                noisy_images=noisy_images,
                sigma=sigma,
                class_labels=class_labels
            )
        
        # Flatten images to compute L2 norm
        noisy_flat = noisy_images.view(batch_size, -1)
        denoised_flat = denoised_images.view(batch_size, -1)
        
        # Compute ||x - x̃||²
        diff_norm_sq = torch.sum((noisy_flat - denoised_flat) ** 2, dim=1)
        
        # Compute ω̂ = ||x - x̃||² / σ³
        omega_hat = diff_norm_sq / (sigma ** 3)
        
        return omega_hat
    
    def estimate(
        self,
        noisy_images: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Alias for forward() for consistency with the computational interface.
        
        Parameters:
        -----------
        noisy_images : torch.Tensor
            Noisy input images of shape (batch_size, C, H, W)
        sigma : torch.Tensor
            Noise levels of shape (batch_size,)
        class_labels : torch.Tensor, optional
            Class labels for conditional models
            
        Returns:
        --------
        omega_hat : torch.Tensor
            Estimated omega_hat values of shape (batch_size,)
        """
        return self.forward(noisy_images, sigma, class_labels)
    
    def eval(self):
        """
        Set estimator to evaluation mode (for consistency with trained models).
        
        Since the EDM model is always in eval mode and has frozen parameters,
        this is effectively a no-op, but provides a consistent interface.
        """
        super().eval()
        self.edm_model.eval()
        return self
    
    def train(self, mode: bool = True):
        """
        Prevent setting estimator to training mode.
        
        Computational estimators cannot be trained, so this raises an error.
        
        Raises:
        -------
        RuntimeError
            Always raised since computational methods cannot be trained
        """
        if mode:
            raise RuntimeError(
                "EDMOmegaEstimator is a computational method and cannot be trained. "
                "It uses a pretrained EDM denoiser for direct omega_hat computation."
            )
        return self

