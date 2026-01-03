"""
Core EDM denoising functions.

This module provides the main denoising functions using pretrained EDM models.
"""

import torch
from typing import Optional, Union

from .utils import _convert_to_tensor


def edm_denoise(
    model: torch.nn.Module,
    noisy_images: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    class_labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Denoise images using a pretrained EDM model.
    
    This function applies the EDM denoiser D(x; σ) to noisy images.
    The caller is responsible for wrapping this in torch.no_grad() if needed.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    noisy_images : torch.Tensor
        Noisy input images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation). Can be a scalar or tensor of shape (batch_size,)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (shape: batch_size,)
        For unconditional models, set to None (default)
        
    Returns:
    --------
    denoised : torch.Tensor
        Denoised images of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from src.external.edm_denoiser import load_pretrained_edm, edm_denoise
    >>> 
    >>> # Load model
    >>> model, config = load_pretrained_edm('cifar10-uncond-ve')
    >>> 
    >>> # Create noisy image
    >>> noisy_img = torch.randn(1, 3, 32, 32).cuda()
    >>> sigma = 2.0
    >>> 
    >>> # Denoise (wrap in no_grad for inference)
    >>> with torch.no_grad():
    >>>     denoised = edm_denoise(model, noisy_img, sigma)
    
    Notes:
    ------
    - The model should already be in eval mode (handled by load_pretrained_edm)
    - For inference, wrap the call in torch.no_grad() context
    - Sigma is automatically broadcast to match batch size if needed
    """
    # Convert sigma to tensor with proper device and dtype
    sigma_tensor = _convert_to_tensor(sigma, noisy_images.device, noisy_images.dtype)
    
    # Replicate sigma for batch if needed
    batch_size = noisy_images.shape[0]
    if sigma_tensor.shape[0] == 1 and batch_size > 1:
        sigma_tensor = sigma_tensor.repeat(batch_size)
    
    # Validate shapes
    if sigma_tensor.shape[0] != batch_size:
        raise ValueError(
            f"Sigma batch size ({sigma_tensor.shape[0]}) must match "
            f"noisy_images batch size ({batch_size}) or be 1"
        )
    
    # Denoise using the model
    denoised = model(noisy_images, sigma_tensor, class_labels=class_labels)
    
    return denoised

