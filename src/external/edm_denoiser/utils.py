"""
Utility functions for EDM denoiser.

This module contains helper functions used by the EDM denoiser.
"""

import torch
from typing import Union


def _convert_to_tensor(
    sigma: Union[float, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert sigma to a tensor with proper device and dtype.
    
    Parameters:
    -----------
    sigma : float or torch.Tensor
        Noise level value
    device : torch.device
        Target device
    dtype : torch.dtype
        Target dtype (default: float32)
        
    Returns:
    --------
    sigma_tensor : torch.Tensor
        Converted sigma tensor of shape (1,)
    """
    if not isinstance(sigma, torch.Tensor):
        sigma_tensor = torch.tensor([sigma], dtype=dtype, device=device)
    else:
        sigma_tensor = sigma.to(device=device, dtype=dtype)
    
    # Ensure it's at least 1D
    if sigma_tensor.dim() == 0:
        sigma_tensor = sigma_tensor.unsqueeze(0)
    
    return sigma_tensor

