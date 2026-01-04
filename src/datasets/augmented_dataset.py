"""
Custom dataset wrapper that applies real-time augmentation.

This module wraps existing datasets and applies augmentation transforms
on-the-fly during data loading.
"""

import torch
from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    """
    Wrapper dataset that applies augmentation transforms in real-time.
    
    This dataset wraps an existing dataset (or Subset) and applies
    augmentation transforms to each sample when __getitem__ is called.
    This ensures that each epoch sees different augmented versions
    of the same images, improving model generalization.
    
    Args:
        base_dataset: The underlying dataset to wrap (can be a Subset)
        augmentation_transform: Augmentation transform to apply (e.g., RandomAugmentation)
                               If None, no augmentation is applied.
    
    Example:
        >>> from src.datasets.augmentation import RandomAugmentation
        >>> aug_transform = RandomAugmentation(
        ...     horizontal_flip=True,
        ...     vertical_flip=True,
        ...     rotation=True,
        ...     gaussian_noise=True
        ... )
        >>> augmented_dataset = AugmentedDataset(train_dataset, aug_transform)
    """
    
    def __init__(self, base_dataset, augmentation_transform=None):
        """
        Initialize the augmented dataset wrapper.
        
        Args:
            base_dataset: Base dataset or Subset to wrap
            augmentation_transform: Transform to apply for augmentation
        """
        self.base_dataset = base_dataset
        self.augmentation_transform = augmentation_transform
    
    def __len__(self):
        """Return the length of the base dataset."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset with augmentation applied.
        
        Args:
            idx (int): Index of the item to retrieve
        
        Returns:
            tuple: (augmented_image, label) where image is augmented if transform is set
        """
        # Get the original item from base dataset
        image, label = self.base_dataset[idx]
        
        # Apply augmentation if transform is provided
        if self.augmentation_transform is not None:
            image = self.augmentation_transform(image)
        
        return image, label
    
    def __repr__(self):
        """String representation of the dataset."""
        base_repr = repr(self.base_dataset)
        aug_repr = repr(self.augmentation_transform) if self.augmentation_transform else "None"
        return f"AugmentedDataset(\n  base_dataset={base_repr},\n  augmentation={aug_repr}\n)"

