"""
Expected Value Omega Estimator.

This module provides a simple computational method for estimating omega_hat using
the statistical expectation of the squared norm of noise, without requiring any
training or pretrained models.

The estimator computes: ω̂ = d / σ
where d is the dimensionality of the image and σ is the noise level.

Mathematical Background:
-----------------------
Since ε ~ N(0, I) with dimension d, we have:
    ||ε||² ~ χ²_d
    E[||ε||²] = d

The ground truth omega_hat is:
    ω̂ = ||ε||² / σ

Therefore, the expected value provides a simple computational estimate:
    E[ω̂] = E[||ε||²] / σ = d / σ

This is the simplest possible estimator that requires no training and no
pretrained models - just the image dimensionality and noise level.

Author: Sigma Score Estimator Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Union, Optional


class ExpectedOmegaEstimator(nn.Module):
    """
    Computational estimator for omega_hat using statistical expectation.
    
    This estimator provides a training-free, model-free method to estimate omega_hat by
    using the expected value of the squared norm of Gaussian noise:
    
        ω̂ = d / σ
    
    where:
        - d is the image dimensionality (e.g., 3*32*32 = 3072 for CIFAR-10)
        - σ is the noise level
    
    This is based on the fact that for ε ~ N(0, I) with dimension d:
        E[||ε||²] = d
    
    And since the ground truth is:
        ω̂ = ||ε||² / σ
    
    The expected value gives us:
        E[ω̂] = E[||ε||²] / σ = d / σ
    
    Parameters:
    -----------
    image_dim : int
        Dimensionality of the input images (default: 3072 for CIFAR-10 3×32×32)
    device : str or torch.device
        Device to run computations on (default: 'cuda' if available, else 'cpu')
        
    Attributes:
    -----------
    image_dim : int
        The dimensionality of images used for estimation
    device : torch.device
        Device where computations are performed
        
    Examples:
    ---------
    >>> import torch
    >>> from src.computational import ExpectedOmegaEstimator
    >>> 
    >>> # Initialize estimator (3×32×32 = 3072 for CIFAR-10)
    >>> estimator = ExpectedOmegaEstimator(image_dim=3072, device='cuda')
    >>> 
    >>> # Generate noisy images (just need sigma, images not actually used)
    >>> batch_size = 4
    >>> noisy_imgs = torch.randn(4, 3, 32, 32).cuda()
    >>> sigma = torch.tensor([2.0, 3.0, 1.5, 2.5]).cuda()
    >>> 
    >>> # Estimate omega_hat
    >>> with torch.no_grad():
    >>>     omega_hat = estimator(noisy_imgs, sigma)
    >>> 
    >>> print(f"Omega hat shape: {omega_hat.shape}")  # torch.Size([4])
    >>> print(f"Values: {omega_hat}")  # [1536, 1024, 2048, 1228.8]
    
    Notes:
    ------
    - This is the simplest possible baseline estimator
    - No training required, no pretrained models required
    - Only requires knowing the image dimensionality
    - Provides the expected (mean) value of omega_hat
    - Actual values will vary around this expectation with variance 2d/σ²
    """
    
    def __init__(
        self,
        image_dim: int = 3072,  # 3×32×32 for CIFAR-10
        device: Optional[Union[str, torch.device]] = None
    ):
        super(ExpectedOmegaEstimator, self).__init__()
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Store image dimensionality
        self.image_dim = image_dim
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('d', torch.tensor(float(image_dim)))
        
        print(f"Expected Omega Estimator initialized on {self.device}")
        print(f"Image dimensionality d = {self.image_dim}")
        print(f"Estimation formula: ω̂ = d / σ = {self.image_dim} / σ")
    
    def forward(
        self,
        noisy_images: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute omega_hat using expected value formula.
        
        This method computes: ω̂ = d / σ
        
        Note: The noisy_images parameter is accepted for API consistency with other
        estimators, but is not actually used in the computation since this method
        only depends on the noise level σ.
        
        Parameters:
        -----------
        noisy_images : torch.Tensor
            Noisy input images of shape (batch_size, C, H, W)
            Note: Not used in computation, only for API consistency
        sigma : torch.Tensor
            Noise levels of shape (batch_size,) or (batch_size, 1)
        class_labels : torch.Tensor, optional
            Class labels (not used, only for API consistency)
            
        Returns:
        --------
        omega_hat : torch.Tensor
            Estimated omega_hat values of shape (batch_size,)
            
        Notes:
        ------
        - The formula is simply: ω̂ = d / σ
        - This provides the expected (mean) value of omega_hat
        - No gradient computation needed (always use with torch.no_grad())
        """
        batch_size = noisy_images.size(0)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Move sigma to same device as model
        if sigma.device != self.device:
            sigma = sigma.to(self.device)
        
        # Compute ω̂ = d / σ
        # Use the registered buffer for d
        omega_hat = self.d / sigma
        
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
            Note: Not used in computation
        sigma : torch.Tensor
            Noise levels of shape (batch_size,)
        class_labels : torch.Tensor, optional
            Class labels (not used)
            
        Returns:
        --------
        omega_hat : torch.Tensor
            Estimated omega_hat values of shape (batch_size,)
        """
        return self.forward(noisy_images, sigma, class_labels)
    
    def eval(self):
        """
        Set estimator to evaluation mode (for consistency with trained models).
        
        Since this is a purely computational method with no learnable parameters,
        this is effectively a no-op, but provides a consistent interface.
        """
        super().eval()
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
                "ExpectedOmegaEstimator is a computational method and cannot be trained. "
                "It uses the statistical expectation E[||ε||²] = d to compute omega_hat "
                "as d / σ."
            )
        return self
    
    def extra_repr(self) -> str:
        """
        Extra representation string for printing.
        """
        return f'image_dim={self.image_dim}, device={self.device}'


