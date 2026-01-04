"""
Real-time data augmentation transforms for training.

This module provides configurable augmentation transforms that are applied
randomly during training to improve model generalization.
"""

import torch
import random
import torch.nn.functional as F


class RandomAugmentation:
    """
    Applies random augmentations to images during training.
    
    Supported augmentations:
    - Horizontal flip
    - Vertical flip
    - Rotation (90, 180, 270 degrees)
    - Gaussian noise
    
    Args:
        horizontal_flip (bool): Whether to apply random horizontal flip
        vertical_flip (bool): Whether to apply random vertical flip
        rotation (bool): Whether to apply random rotation (90, 180, 270 degrees)
        gaussian_noise (bool): Whether to add Gaussian noise
        gaussian_noise_std (float): Standard deviation for Gaussian noise
        flip_prob (float): Probability of applying flip operations (default: 0.5)
        rotation_prob (float): Probability of applying rotation (default: 0.5)
        noise_prob (float): Probability of adding Gaussian noise (default: 0.5)
    """
    
    def __init__(
        self,
        horizontal_flip=True,
        vertical_flip=True,
        rotation=True,
        gaussian_noise=True,
        gaussian_noise_std=0.1,
        flip_prob=0.5,
        rotation_prob=0.5,
        noise_prob=0.5
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.gaussian_noise = gaussian_noise
        self.gaussian_noise_std = gaussian_noise_std
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
        self.noise_prob = noise_prob
    
    def __call__(self, img):
        """
        Apply random augmentations to an image.
        
        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W)
        
        Returns:
            torch.Tensor: Augmented image tensor
        """
        # Apply horizontal flip
        if self.horizontal_flip and random.random() < self.flip_prob:
            img = torch.flip(img, dims=[2])  # Flip along width dimension
        
        # Apply vertical flip
        if self.vertical_flip and random.random() < self.flip_prob:
            img = torch.flip(img, dims=[1])  # Flip along height dimension
        
        # Apply rotation (90, 180, or 270 degrees)
        if self.rotation and random.random() < self.rotation_prob:
            # Randomly select number of 90-degree rotations (1, 2, or 3)
            k = random.choice([1, 2, 3])
            img = torch.rot90(img, k=k, dims=[1, 2])
        
        # Add Gaussian noise
        if self.gaussian_noise and random.random() < self.noise_prob:
            noise = torch.randn_like(img) * self.gaussian_noise_std
            img = img + noise
            # Clamp to maintain valid range (assuming normalized images)
            img = torch.clamp(img, min=-1.0, max=1.0)
        
        return img
    
    def __repr__(self):
        """String representation of augmentation configuration."""
        aug_list = []
        if self.horizontal_flip:
            aug_list.append(f"HorizontalFlip(p={self.flip_prob})")
        if self.vertical_flip:
            aug_list.append(f"VerticalFlip(p={self.flip_prob})")
        if self.rotation:
            aug_list.append(f"Rotation90/180/270(p={self.rotation_prob})")
        if self.gaussian_noise:
            aug_list.append(f"GaussianNoise(std={self.gaussian_noise_std}, p={self.noise_prob})")
        
        if not aug_list:
            return "RandomAugmentation(None)"
        
        return "RandomAugmentation(\n  " + "\n  ".join(aug_list) + "\n)"


def create_augmentation_from_config(aug_config):
    """
    Create a RandomAugmentation instance from configuration dictionary.
    
    Args:
        aug_config (dict): Configuration dictionary with augmentation settings
    
    Returns:
        RandomAugmentation or None: Augmentation transform or None if disabled
    
    Example config:
        {
            'enabled': True,
            'horizontal_flip': True,
            'vertical_flip': True,
            'rotation': True,
            'gaussian_noise': True,
            'gaussian_noise_std': 0.1,
            'flip_prob': 0.5,
            'rotation_prob': 0.5,
            'noise_prob': 0.5
        }
    """
    if aug_config is None or not aug_config.get('enabled', False):
        return None
    
    return RandomAugmentation(
        horizontal_flip=aug_config.get('horizontal_flip', True),
        vertical_flip=aug_config.get('vertical_flip', True),
        rotation=aug_config.get('rotation', True),
        gaussian_noise=aug_config.get('gaussian_noise', True),
        gaussian_noise_std=aug_config.get('gaussian_noise_std', 0.1),
        flip_prob=aug_config.get('flip_prob', 0.5),
        rotation_prob=aug_config.get('rotation_prob', 0.5),
        noise_prob=aug_config.get('noise_prob', 0.5)
    )

