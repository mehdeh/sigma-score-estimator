"""
Hybrid Omega Estimator.

This module provides a hybrid computational method that combines the strengths of
both EDM-based and Expected Value methods by using a sigma threshold:
- For σ < threshold: Use expected value method (d/σ)
- For σ ≥ threshold: Use EDM denoiser method (||x - x̃||²/σ³)

The rationale is that:
- Expected value method works better for small sigma (high SNR)
- EDM denoiser method works better for large sigma (low SNR)

This hybrid approach leverages the best of both methods.

Author: Sigma Score Estimator Team
Date: 2025
"""

import torch
import torch.nn as nn
from typing import Union, Optional

from .edm_estimator import EDMOmegaEstimator
from .expected_estimator import ExpectedOmegaEstimator


class HybridOmegaEstimator(nn.Module):
    """
    Hybrid computational estimator combining EDM and Expected Value methods.
    
    This estimator provides an adaptive training-free method that selects between
    two computational approaches based on the noise level:
    
    - For σ < sigma_threshold: ω̂ = d / σ (Expected Value method)
    - For σ ≥ sigma_threshold: ω̂ = ||x - D_EDM(x,σ)||² / σ³ (EDM method)
    
    The threshold is configurable and allows leveraging the strengths of each method:
    - Expected value is more accurate for small σ (high signal-to-noise ratio)
    - EDM denoiser is more accurate for large σ (low signal-to-noise ratio)
    
    Parameters:
    -----------
    sigma_threshold : float
        Threshold value for switching between methods (default: 1.0)
        - Below this: use expected value d/σ
        - Above this: use EDM denoiser
    edm_model_name : str
        Name of the pretrained EDM model to use (default: 'cifar10-uncond-ve')
    image_dim : int
        Dimensionality of the input images (default: 3072 for CIFAR-10)
    device : str or torch.device
        Device to run computations on (default: 'cuda' if available, else 'cpu')
        
    Attributes:
    -----------
    sigma_threshold : float
        The threshold value for method selection
    edm_estimator : EDMOmegaEstimator
        EDM-based estimator for large sigma
    expected_estimator : ExpectedOmegaEstimator
        Expected value estimator for small sigma
    device : torch.device
        Device where computations are performed
        
    Examples:
    ---------
    >>> import torch
    >>> from src.computational import HybridOmegaEstimator
    >>> 
    >>> # Initialize hybrid estimator with threshold=1.0
    >>> estimator = HybridOmegaEstimator(
    >>>     sigma_threshold=1.0,
    >>>     edm_model_name='cifar10-uncond-ve',
    >>>     image_dim=3072,
    >>>     device='cuda'
    >>> )
    >>> 
    >>> # Generate noisy images with mixed sigma values
    >>> batch_size = 4
    >>> noisy_imgs = torch.randn(4, 3, 32, 32).cuda()
    >>> sigma = torch.tensor([0.5, 1.5, 0.8, 2.0]).cuda()  # Mixed small and large
    >>> 
    >>> # Estimate omega_hat (automatically selects method per sample)
    >>> with torch.no_grad():
    >>>     omega_hat = estimator(noisy_imgs, sigma)
    >>> 
    >>> # First and third samples use expected value (σ < 1.0)
    >>> # Second and fourth samples use EDM (σ ≥ 1.0)
    >>> print(f"Omega hat shape: {omega_hat.shape}")  # torch.Size([4])
    
    Notes:
    ------
    - Per-sample method selection: Each sample in a batch can use a different method
    - Efficient batching: Splits batch into two groups for efficient computation
    - Automatic switching: No manual intervention needed, threshold-based
    - Best of both worlds: Combines accuracy benefits of both approaches
    """
    
    def __init__(
        self,
        sigma_threshold: float = 1.0,
        edm_model_name: str = 'cifar10-uncond-ve',
        image_dim: int = 3072,
        device: Optional[Union[str, torch.device]] = None
    ):
        super(HybridOmegaEstimator, self).__init__()
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Store threshold
        self.sigma_threshold = sigma_threshold
        self.image_dim = image_dim
        
        # Initialize both estimators
        print(f"Initializing Hybrid Omega Estimator on {self.device}")
        print(f"Sigma threshold: {sigma_threshold}")
        print(f"  - For σ < {sigma_threshold}: Use Expected Value (d/σ)")
        print(f"  - For σ ≥ {sigma_threshold}: Use EDM Denoiser (||x-x̃||²/σ³)")
        
        # EDM estimator for large sigma
        self.edm_estimator = EDMOmegaEstimator(
            model_name=edm_model_name,
            device=self.device
        )
        
        # Expected value estimator for small sigma
        self.expected_estimator = ExpectedOmegaEstimator(
            image_dim=image_dim,
            device=self.device
        )
        
        print(f"Hybrid Omega Estimator initialized successfully")
    
    def forward(
        self,
        noisy_images: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute omega_hat using hybrid method.
        
        This method:
        1. Splits samples based on sigma threshold
        2. Applies expected value method to small sigma samples
        3. Applies EDM method to large sigma samples
        4. Combines results maintaining original order
        
        Parameters:
        -----------
        noisy_images : torch.Tensor
            Noisy input images of shape (batch_size, C, H, W)
        sigma : torch.Tensor
            Noise levels of shape (batch_size,) or (batch_size, 1)
        class_labels : torch.Tensor, optional
            Class labels for conditional EDM models (shape: batch_size,)
            
        Returns:
        --------
        omega_hat : torch.Tensor
            Estimated omega_hat values of shape (batch_size,)
            
        Notes:
        ------
        - Automatically selects method per sample based on sigma
        - Maintains batch order in output
        - Efficient: Only runs EDM on samples that need it
        """
        batch_size = noisy_images.size(0)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Move to device if needed
        if sigma.device != self.device:
            sigma = sigma.to(self.device)
        if noisy_images.device != self.device:
            noisy_images = noisy_images.to(self.device)
        
        # Create mask for small vs large sigma
        small_sigma_mask = sigma < self.sigma_threshold
        large_sigma_mask = ~small_sigma_mask
        
        # Initialize output tensor
        omega_hat = torch.zeros(batch_size, device=self.device, dtype=sigma.dtype)
        
        # Process small sigma samples with expected value method
        if small_sigma_mask.any():
            small_indices = torch.where(small_sigma_mask)[0]
            small_sigma = sigma[small_indices]
            small_images = noisy_images[small_indices]
            
            # Expected value: d/σ (doesn't actually use images)
            omega_hat[small_indices] = self.expected_estimator(
                small_images,
                small_sigma,
                class_labels[small_indices] if class_labels is not None else None
            )
        
        # Process large sigma samples with EDM method
        if large_sigma_mask.any():
            large_indices = torch.where(large_sigma_mask)[0]
            large_sigma = sigma[large_indices]
            large_images = noisy_images[large_indices]
            
            # EDM: ||x - D_EDM(x,σ)||²/σ³
            omega_hat[large_indices] = self.edm_estimator(
                large_images,
                large_sigma,
                class_labels[large_indices] if class_labels is not None else None
            )
        
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
        Set estimator to evaluation mode.
        
        Ensures both sub-estimators are in eval mode.
        """
        super().eval()
        self.edm_estimator.eval()
        self.expected_estimator.eval()
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
                "HybridOmegaEstimator is a computational method and cannot be trained. "
                "It combines pretrained EDM denoiser and statistical expectation."
            )
        return self
    
    def extra_repr(self) -> str:
        """
        Extra representation string for printing.
        """
        return (
            f'sigma_threshold={self.sigma_threshold}, '
            f'image_dim={self.image_dim}, '
            f'device={self.device}'
        )
    
    def get_method_statistics(self, sigma: torch.Tensor) -> dict:
        """
        Get statistics about which method would be used for given sigma values.
        
        Useful for analysis and understanding the hybrid behavior.
        
        Parameters:
        -----------
        sigma : torch.Tensor
            Noise levels to analyze
            
        Returns:
        --------
        stats : dict
            Dictionary containing:
            - num_expected: Number of samples using expected value method
            - num_edm: Number of samples using EDM method
            - fraction_expected: Fraction using expected value
            - fraction_edm: Fraction using EDM
        """
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        small_mask = sigma < self.sigma_threshold
        num_expected = small_mask.sum().item()
        num_edm = (~small_mask).sum().item()
        total = len(sigma)
        
        return {
            'num_expected': num_expected,
            'num_edm': num_edm,
            'fraction_expected': num_expected / total if total > 0 else 0.0,
            'fraction_edm': num_edm / total if total > 0 else 0.0,
            'sigma_threshold': self.sigma_threshold,
        }


