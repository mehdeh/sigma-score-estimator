"""
Score function computation and gradient ascent denoising.

This module implements score-based denoising methods using EDM models.
"""

import torch
from typing import Optional, Union, Tuple, List


def compute_score_gradient(
    model: torch.nn.Module,
    x: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    class_labels: Optional[torch.Tensor] = None,
    use_float64: bool = False
) -> torch.Tensor:
    """
    Compute the score function (gradient of log probability density) at x.
    
    The score is defined as: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    where D(x; σ) is the denoiser output. This is derived from Tweedie's formula.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x : torch.Tensor
        Input images of shape (batch_size, C, H, W)
        Note: Does NOT need requires_grad=True (computed analytically)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (default: None)
    use_float64 : bool
        If True, use double precision (float64) for better numerical stability (default: False)
        
    Returns:
    --------
    score : torch.Tensor
        Score function (gradient direction) of shape (batch_size, C, H, W)
        
    Examples:
    ---------
    >>> import torch
    >>> from src.external.edm_denoiser import load_pretrained_edm, compute_score_gradient
    >>> 
    >>> # Load model
    >>> model, config = load_pretrained_edm('cifar10-uncond-ve')
    >>> 
    >>> # Compute score at a point
    >>> x = torch.randn(1, 3, 32, 32).cuda()
    >>> sigma = 2.0
    >>> score = compute_score_gradient(model, x, sigma)
    >>> 
    >>> # Use for gradient ascent
    >>> learning_rate = 1.0
    >>> x_updated = x + learning_rate * score
    
    Notes:
    ------
    - The score is computed analytically using the denoiser output
    - No backpropagation is needed through the denoiser
    - For gradient ascent, use this in a loop to iteratively move x
    - Reference: EDM paper (Karras et al., 2022), Equation 4
    """
    # Determine working dtype
    working_dtype = torch.float64 if use_float64 else x.dtype
    original_dtype = x.dtype
    
    # Convert x to working dtype if needed
    x_work = x.to(working_dtype) if use_float64 else x
    
    # Convert sigma to tensor with proper device and working dtype
    if not isinstance(sigma, torch.Tensor):
        sigma_tensor = torch.tensor([sigma], dtype=working_dtype, device=x.device)
    else:
        sigma_tensor = sigma.to(device=x.device, dtype=working_dtype)
        if sigma_tensor.dim() == 0:
            sigma_tensor = sigma_tensor.unsqueeze(0)
    
    # Get denoised output (no gradient needed)
    with torch.no_grad():
        denoised = model(x_work, sigma_tensor, class_labels=class_labels)
        denoised = denoised.to(working_dtype)
    
    # Compute score: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    sigma_sq = sigma_tensor ** 2
    score = -(x_work - denoised) / sigma_sq
    
    # Convert back to original dtype if needed
    if use_float64 and original_dtype != torch.float64:
        score = score.to(original_dtype)
    
    return score


def gradient_ascent_denoise(
    model: torch.nn.Module,
    x_init: torch.Tensor,
    sigma: Union[float, torch.Tensor],
    num_steps: int = 10,
    lr: float = 1.0,
    class_labels: Optional[torch.Tensor] = None,
    return_trajectory: bool = False,
    use_float64: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """
    Perform gradient ascent on the log probability to denoise images.
    
    This iteratively moves x towards higher probability regions by following
    the score function: x_{t+1} = x_t + lr * ∇_x log p(x_t; σ)
    
    This implements gradient ascent at a fixed noise level for denoising.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Pretrained EDM denoising model
    x_init : torch.Tensor
        Initial noisy images of shape (batch_size, C, H, W)
    sigma : float or torch.Tensor
        Noise level (standard deviation)
    num_steps : int
        Number of gradient ascent iterations (default: 10)
    lr : float
        Learning rate for gradient updates (default: 1.0)
    class_labels : torch.Tensor, optional
        Class labels for conditional models (default: None)
    return_trajectory : bool
        If True, return the full trajectory of intermediate images (default: False)
    use_float64 : bool
        If True, use double precision (float64) for better numerical stability (default: True)
        
    Returns:
    --------
    x_final : torch.Tensor
        Denoised images after gradient ascent
    trajectory : list of torch.Tensor (optional)
        List of intermediate images at each step (only if return_trajectory=True)
        
    Examples:
    ---------
    >>> import torch
    >>> from src.external.edm_denoiser import load_pretrained_edm, gradient_ascent_denoise
    >>> 
    >>> # Setup
    >>> model, config = load_pretrained_edm('cifar10-uncond-ve')
    >>> noisy_img = torch.randn(1, 3, 32, 32).cuda()
    >>> 
    >>> # Gradient ascent denoising
    >>> denoised, trajectory = gradient_ascent_denoise(
    >>>     model, noisy_img, sigma=3.0, num_steps=10, lr=1.0, return_trajectory=True
    >>> )
    >>> 
    >>> # Or without trajectory
    >>> denoised = gradient_ascent_denoise(model, noisy_img, sigma=3.0, num_steps=10)
    
    Notes:
    ------
    - This is different from the full EDM sampler which uses a noise schedule
    - Here we denoise at a fixed noise level using gradient ascent
    - Uses float64 by default for better numerical precision during optimization
    - The score function is: ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
    """
    # Determine working dtype
    working_dtype = torch.float64 if use_float64 else x_init.dtype
    original_dtype = x_init.dtype
    
    # Convert x to working dtype
    x_current = x_init.clone().detach().to(working_dtype)
    trajectory = [x_current.clone()] if return_trajectory else []
    
    # Convert sigma to tensor with proper device and working dtype
    if not isinstance(sigma, torch.Tensor):
        sigma_tensor = torch.tensor([sigma], dtype=working_dtype, device=x_init.device)
    else:
        sigma_tensor = sigma.to(device=x_init.device, dtype=working_dtype)
        if sigma_tensor.dim() == 0:
            sigma_tensor = sigma_tensor.unsqueeze(0)
    
    # Precompute sigma squared (keep as 1D tensor for natural broadcasting)
    sigma_sq = sigma_tensor ** 2
    
    for step in range(num_steps):
        # Get denoised output from model
        with torch.no_grad():
            # Model expects specific dtype, convert back if needed
            denoised = model(x_current, sigma_tensor, class_labels=class_labels)
            denoised = denoised.to(working_dtype)
            
            # Compute gradient of log probability
            # ∇_x log p(x; σ) = -(x - D(x; σ)) / σ²
            grad_log_prob = -(x_current - denoised) / sigma_sq
            
            # Gradient ascent step
            x_current = x_current + lr * grad_log_prob
            
        if return_trajectory:
            trajectory.append(x_current.clone())
    
    # Convert back to original dtype if needed
    if use_float64 and original_dtype != torch.float64:
        x_current = x_current.to(original_dtype)
        if return_trajectory:
            trajectory = [x.to(original_dtype) for x in trajectory]
    
    if return_trajectory:
        return x_current, trajectory
    else:
        return x_current

