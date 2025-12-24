"""
Shared evaluation utilities for omega estimator models.

This module contains common functions used by both OmegaTrainer and OmegaEvaluator
to avoid code duplication and maintain consistency.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def compute_omega_hat_target(clean_images, noisy_images, sigma):
    """
    Compute ground truth omega_hat for evaluation.
    
    Computes: ω̂_target = ||x - x̃||² / σ³
    
    This is mathematically equivalent to: ||ε||² / σ
    where ε = (x̃ - x) / σ
    
    All evaluation metrics are computed in omega_hat space after transformation.
    This provides a consistent evaluation metric across all loss types.
    
    Args:
        clean_images (Tensor): Clean images, shape (batch_size, C, H, W)
        noisy_images (Tensor): Noisy images, shape (batch_size, C, H, W)
        sigma (Tensor): Noise levels, shape (batch_size,)
    
    Returns:
        Tensor: Ground truth omega_hat = ||x - x̃||² / σ³
    """
    batch_size = clean_images.size(0)
    
    # Flatten images
    clean_flat = clean_images.view(batch_size, -1)
    noisy_flat = noisy_images.view(batch_size, -1)
    
    # Ensure sigma is 1D
    if sigma.dim() > 1:
        sigma = sigma.squeeze()
    
    # Compute epsilon = (x_tilde - x) / sigma
    epsilon = (noisy_flat - clean_flat) / sigma.view(-1, 1)
    
    # Compute ||epsilon||²
    epsilon_norm_sq = torch.sum(epsilon ** 2, dim=1)
    
    # Ground truth omega_hat = ||epsilon||² / sigma
    # This equals ||x - x̃||² / σ³
    omega_hat_target = epsilon_norm_sq / sigma
    
    return omega_hat_target


def compute_raw_target(clean_images, noisy_images, sigma, loss_type):
    """
    Compute ground truth in the same space as raw model output (before transformation).
    
    For omega_chi_zscore: Computes z-score = (||ε||² - d) / sqrt(2*d)
    For other losses: Same as omega_hat target (no transformation)
    
    Args:
        clean_images (Tensor): Clean images, shape (batch_size, C, H, W)
        noisy_images (Tensor): Noisy images, shape (batch_size, C, H, W)
        sigma (Tensor): Noise levels, shape (batch_size,)
        loss_type (str): Type of loss function being used
    
    Returns:
        Tensor: Ground truth in raw output space
    """
    batch_size = clean_images.size(0)
    
    # Flatten images
    clean_flat = clean_images.view(batch_size, -1)
    noisy_flat = noisy_images.view(batch_size, -1)
    d = clean_flat.size(1)  # Image dimensionality
    
    # Ensure sigma is 1D
    if sigma.dim() > 1:
        sigma = sigma.squeeze()
    
    # Compute epsilon = (x_tilde - x) / sigma
    epsilon = (noisy_flat - clean_flat) / sigma.view(-1, 1)
    
    # Compute ||epsilon||²
    epsilon_norm_sq = torch.sum(epsilon ** 2, dim=1)
    
    # For omega_chi_zscore, compute z-score
    if loss_type == 'omega_chi_zscore':
        # z-score = (||epsilon||² - d) / sqrt(2*d)
        sqrt_2d = torch.sqrt(torch.tensor(2.0 * d, device=epsilon_norm_sq.device))
        raw_target = (epsilon_norm_sq - d) / sqrt_2d
    else:
        # For other losses, raw output is already omega_hat
        raw_target = epsilon_norm_sq / sigma
    
    return raw_target


def compute_evaluation_metrics(predictions, targets, losses):
    """
    Compute evaluation metrics from predictions and targets.
    
    This function computes standard regression metrics used for evaluating
    the omega estimator model performance.
    
    Args:
        predictions (np.ndarray): Model predictions
        targets (np.ndarray): Ground truth targets
        losses (np.ndarray): Loss values for each sample
    
    Returns:
        dict: Dictionary containing evaluation metrics:
            - mse: Mean squared error
            - mae: Mean absolute error
            - r2_score: R-squared score
            - avg_loss: Average loss
            - mean_relative_error: Mean of relative errors
            - median_relative_error: Median of relative errors
            - num_samples: Number of samples evaluated
    """
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    avg_loss = np.mean(losses)
    
    # Compute relative error metrics
    relative_errors = np.abs(predictions - targets) / (np.abs(targets) + 1e-8)
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2_score': float(r2),
        'avg_loss': float(avg_loss),
        'mean_relative_error': float(mean_relative_error),
        'median_relative_error': float(median_relative_error),
        'num_samples': len(predictions),
    }
    
    return metrics


def log_evaluation_metrics(logger, metrics, phase='Evaluation'):
    """
    Log evaluation metrics to logger.
    
    Args:
        logger: Logger instance
        metrics (dict): Dictionary of metrics to log
        phase (str): Phase name (e.g., 'Training', 'Validation', 'Test')
    """
    logger.info(f"{phase} Results:")
    logger.info(f"  Number of samples: {metrics['num_samples']}")
    logger.info(f"  MSE: {metrics['mse']:.6f}")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  R² Score: {metrics['r2_score']:.6f}")
    logger.info(f"  Average Loss: {metrics['avg_loss']:.6f}")
    logger.info(f"  Mean Relative Error: {metrics['mean_relative_error']:.4f}")
    logger.info(f"  Median Relative Error: {metrics['median_relative_error']:.4f}")

