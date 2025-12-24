"""
Noise generation utilities for training the omega estimator.
Provides multiple strategies for sampling noise levels and adding Gaussian noise to images.
"""

import torch
import numpy as np


class NoiseGenerator:
    """
    Generator for sampling noise levels and adding Gaussian noise to images.
    
    Supports multiple sampling strategies:
    - 'uniform': Uniform sampling in [sigma_min, sigma_max]
    - 'log_uniform': Log-uniform sampling (uniform in log-space)
    - 'select_batch': Select from pre-defined batch of values
    
    Args:
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        strategy (str): Sampling strategy ('uniform', 'log_uniform', 'select_batch')
        num_samples (int): Number of pre-filtered samples for 'select_batch' strategy
        device (str or torch.device): Device to create tensors on
    """
    
    def __init__(
        self,
        sigma_min=0.01,
        sigma_max=10.0,
        strategy='uniform',
        num_samples=100,
        device='cuda'
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.strategy = strategy
        self.num_samples = num_samples
        self.device = device
        
        # Pre-generate filtered samples for select_batch strategy
        if strategy == 'select_batch':
            self.filtered_samples = self._generate_filtered_samples()
    
    def _generate_filtered_samples(self):
        """
        Generate a set of filtered sigma values for batch selection.
        
        Returns:
            Tensor of pre-sampled sigma values
        """
        # Generate uniformly distributed samples in log-space for better coverage
        log_min = np.log(self.sigma_min)
        log_max = np.log(self.sigma_max)
        log_samples = np.linspace(log_min, log_max, self.num_samples)
        samples = np.exp(log_samples)
        return torch.tensor(samples, dtype=torch.float32, device=self.device)
    
    def generate_sigma(self, batch_size):
        """
        Generate noise level values for a batch.
        
        Args:
            batch_size (int): Number of sigma values to generate
        
        Returns:
            Tensor of shape (batch_size,) containing noise levels
        """
        if self.strategy == 'uniform':
            return self._uniform_sampling(batch_size)
        elif self.strategy == 'log_uniform':
            return self._log_uniform_sampling(batch_size)
        elif self.strategy == 'select_batch':
            return self._select_uniform_batch(batch_size)
        else:
            raise ValueError(
                f"Unknown noise sampling strategy: {self.strategy}. "
                f"Expected 'uniform', 'log_uniform', or 'select_batch'."
            )
    
    def _uniform_sampling(self, batch_size):
        """
        Uniform sampling in [sigma_min, sigma_max].
        
        Args:
            batch_size (int): Number of samples
        
        Returns:
            Tensor of uniformly sampled sigma values
        """
        sigma = torch.rand(batch_size, device=self.device)
        sigma = sigma * (self.sigma_max - self.sigma_min) + self.sigma_min
        return sigma
    
    def _log_uniform_sampling(self, batch_size):
        """
        Log-uniform sampling (uniform in log-space).
        This provides more samples at lower sigma values.
        
        Args:
            batch_size (int): Number of samples
        
        Returns:
            Tensor of log-uniformly sampled sigma values
        """
        log_min = np.log(self.sigma_min)
        log_max = np.log(self.sigma_max)
        
        # Sample uniformly in log-space
        log_sigma = torch.rand(batch_size, device=self.device)
        log_sigma = log_sigma * (log_max - log_min) + log_min
        
        # Convert back to linear space
        sigma = torch.exp(log_sigma)
        return sigma
    
    def _select_uniform_batch(self, batch_size):
        """
        Select uniformly from pre-filtered samples.
        
        Args:
            batch_size (int): Number of samples
        
        Returns:
            Tensor of sigma values selected from filtered samples
        """
        # Randomly select indices from filtered samples
        indices = torch.randint(
            0, len(self.filtered_samples), (batch_size,), device=self.device
        )
        sigma = self.filtered_samples[indices]
        return sigma
    
    def add_noise(self, images, sigma):
        """
        Add Gaussian noise to images.
        
        The noise is sampled from N(0, sigma^2 * I), where sigma is specified per image.
        
        Args:
            images (Tensor): Clean images of shape (batch_size, C, H, W)
            sigma (Tensor): Noise levels of shape (batch_size,)
        
        Returns:
            tuple: (noisy_images, noise)
                - noisy_images (Tensor): Images with added noise
                - noise (Tensor): The noise that was added
        """
        # Generate noise: N(0, sigma^2 * I)
        noise = torch.randn_like(images)
        
        # Scale noise by sigma for each image in the batch
        # sigma.view(-1, 1, 1, 1) broadcasts to (batch_size, 1, 1, 1)
        noise = noise * sigma.view(-1, 1, 1, 1)
        
        # Add noise to images
        noisy_images = images + noise
        
        return noisy_images, noise
    
    def compute_noise_norm_squared(self, noise):
        """
        Compute the L2 norm squared of noise for each image in the batch.
        
        This is used as the target for some loss functions: ||noise||^2
        
        Args:
            noise (Tensor): Noise tensor of shape (batch_size, C, H, W)
        
        Returns:
            Tensor of shape (batch_size,) containing ||noise_i||^2 for each image
        """
        # Flatten spatial dimensions and compute squared L2 norm
        noise_flat = noise.view(noise.size(0), -1)
        norm_squared = (noise_flat ** 2).sum(dim=1)
        return norm_squared


def generate_torch_random_vector(size, sigma_max=10.0, device='cuda'):
    """
    Legacy function for generating random sigma values.
    Uses uniform sampling in [0, sigma_max].
    
    Args:
        size (int): Number of values to generate
        sigma_max (float): Maximum sigma value
        device (str or torch.device): Device to create tensor on
    
    Returns:
        Tensor of shape (size,) with random sigma values
    """
    return torch.rand(size, device=device) * sigma_max


def select_uniform_batch(filtered_samples, batch_size, device='cuda'):
    """
    Legacy function for selecting from filtered samples.
    
    Args:
        filtered_samples (Tensor or array): Pre-computed sigma values
        batch_size (int): Number of samples to select
        device (str or torch.device): Device to create tensor on
    
    Returns:
        Tensor of shape (batch_size,) with selected sigma values
    """
    if not isinstance(filtered_samples, torch.Tensor):
        filtered_samples = torch.tensor(
            filtered_samples, dtype=torch.float32, device=device
        )
    
    indices = torch.randint(0, len(filtered_samples), (batch_size,), device=device)
    return filtered_samples[indices]


