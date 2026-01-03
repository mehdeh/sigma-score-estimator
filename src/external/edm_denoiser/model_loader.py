"""
EDM model loading utilities.

This module handles downloading and loading pretrained EDM models.
Includes download utilities and model management functions.
"""

import torch
import pickle
import os
import urllib.request
from typing import Optional, Union, Tuple
from tqdm import tqdm


# ========== Download Utilities ==========

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, blocks=1, block_size=1, total_size=None):
        """Update progress bar for urllib.request.urlretrieve."""
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, destination: str) -> None:
    """
    Download a file from URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    destination : str
        Local path to save the downloaded file
        
    Examples:
    ---------
    >>> from src.external.edm_denoiser.model_loader import download_file
    >>> 
    >>> url = "https://example.com/model.pkl"
    >>> dest = "./models/model.pkl"
    >>> download_file(url, dest)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    print(f"Downloading from {url}")
    print(f"Saving to {destination}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Download") as progress_bar:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=progress_bar.update_to
        )
    
    print(f"✓ Download completed: {destination}")


def ensure_model_downloaded(model_path: str, url: str) -> bool:
    """
    Ensure a model file exists, downloading if necessary.
    
    Parameters:
    -----------
    model_path : str
        Local path where model should exist
    url : str
        URL to download from if model doesn't exist
        
    Returns:
    --------
    success : bool
        True if model exists or was successfully downloaded
        
    Examples:
    ---------
    >>> from src.external.edm_denoiser.model_loader import ensure_model_downloaded
    >>> 
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    >>> 
    >>> if ensure_model_downloaded(model_path, url):
    >>>     print("Model ready to use")
    """
    if os.path.exists(model_path):
        print(f"✓ Model found at: {model_path}")
        return True
    
    print(f"Model not found at: {model_path}")
    
    try:
        download_file(url, model_path)
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False


# ========== Model Loading Functions ==========

def load_edm_model(
    model_path: str,
    url: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None
) -> torch.nn.Module:
    """
    Load a pretrained EDM denoising model from a local path or URL.
    
    If the model file does not exist locally and a URL is provided,
    it will attempt to download the model from the URL and save it locally.
    
    Parameters:
    -----------
    model_path : str
        Local path where the model file should be stored and loaded from
    url : str, optional
        URL to download the model file if it does not already exist locally
    device : torch.device or str, optional
        Device to which the model will be moved (default: 'cuda' if available, else 'cpu')
        
    Returns:
    --------
    net : torch.nn.Module
        Loaded EDM denoising model in eval mode
        
    Examples:
    ---------
    >>> import torch
    >>> from src.external.edm_denoiser import load_edm_model
    >>> 
    >>> # Load pretrained CIFAR-10 model
    >>> model_path = "./pretrain_models/edm-cifar10-32x32-uncond-ve.pkl"
    >>> model_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl"
    >>> 
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> model = load_edm_model(model_path, model_url, device)
    >>> 
    >>> # Use the model for denoising
    >>> noisy_img = torch.randn(1, 3, 32, 32).to(device)
    >>> sigma = torch.tensor([2.0]).to(device)
    >>> denoised = model(noisy_img, sigma, class_labels=None)
    """
    # Set default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    # Check if the model file already exists, download if needed
    if not os.path.exists(model_path) and url is not None:
        print(f"Model not found at {model_path}. Downloading...")
        if not ensure_model_downloaded(model_path, url):
            raise RuntimeError(f"Failed to download model from {url}")
    
    # Load the model from disk
    if os.path.exists(model_path):
        print(f"Loading EDM model from: {model_path}")
        with open(model_path, "rb") as f:
            net = pickle.load(f)['ema'].to(device)
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path} and no URL provided for download."
        )
    
    net.eval()
    return net


# Predefined model configurations
EDM_PRETRAINED_MODELS = {
    'cifar10-uncond-vp': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-vp.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ddpmpp'
    },
    'cifar10-uncond-ve': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ncsnpp'
    },
    'cifar10-cond-vp': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-vp.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ddpmpp'
    },
    'cifar10-cond-ve': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ncsnpp'
    },
    # Legacy aliases for backward compatibility
    'cifar10-uncond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-uncond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': False,
        'architecture': 'ncsnpp'
    },
    'cifar10-cond': {
        'url': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-ve.pkl',
        'path': './pretrain_models/edm-cifar10-32x32-cond-ve.pkl',
        'resolution': 32,
        'channels': 3,
        'conditional': True,
        'architecture': 'ncsnpp'
    }
}


def load_pretrained_edm(
    model_name: str = 'cifar10-uncond',
    device: Optional[Union[torch.device, str]] = None
) -> Tuple[torch.nn.Module, dict]:
    """
    Load a pretrained EDM model by name.
    
    This is a convenience function that loads commonly used pretrained models
    from the official EDM repository.
    
    Parameters:
    -----------
    model_name : str
        Name of the pretrained model (default: 'cifar10-uncond')
        Available models:
            - 'cifar10-uncond-vp': Unconditional CIFAR-10, VP parameterization
            - 'cifar10-uncond-ve': Unconditional CIFAR-10, VE parameterization
            - 'cifar10-cond-vp': Conditional CIFAR-10, VP parameterization
            - 'cifar10-cond-ve': Conditional CIFAR-10, VE parameterization
            - 'cifar10-uncond': Alias for 'cifar10-uncond-ve'
            - 'cifar10-cond': Alias for 'cifar10-cond-ve'
    device : torch.device or str, optional
        Device to load the model on (default: cuda if available, else cpu)
        
    Returns:
    --------
    model : torch.nn.Module
        Loaded pretrained model in eval mode
    config : dict
        Model configuration dictionary containing:
            - url: Download URL
            - path: Local path
            - resolution: Image resolution
            - channels: Number of channels
            - conditional: Whether the model is conditional
            - architecture: Model architecture type
        
    Examples:
    ---------
    >>> from src.external.edm_denoiser import load_pretrained_edm
    >>> 
    >>> # Load unconditional model (VE parameterization)
    >>> model, config = load_pretrained_edm('cifar10-uncond-ve')
    >>> print(f"Resolution: {config['resolution']}")
    >>> print(f"Conditional: {config['conditional']}")
    >>> 
    >>> # Load conditional model (VP parameterization)
    >>> model, config = load_pretrained_edm('cifar10-cond-vp')
    >>> 
    >>> # Use with specific device
    >>> model, config = load_pretrained_edm('cifar10-uncond', device='cuda:0')
    
    Raises:
    -------
    ValueError
        If the model_name is not recognized
        
    Notes:
    ------
    - VP (Variance Preserving) and VE (Variance Exploding) refer to different
      noise schedules and model architectures
    - For most use cases, the VE models (default) work well
    - Requires ~200-300MB download on first use
    """
    if model_name not in EDM_PRETRAINED_MODELS:
        available = list(set(EDM_PRETRAINED_MODELS.keys()) - {'cifar10-uncond', 'cifar10-cond'})
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Available models: {available}"
        )
    
    config = EDM_PRETRAINED_MODELS[model_name].copy()
    model = load_edm_model(config['path'], config['url'], device)
    
    return model, config

