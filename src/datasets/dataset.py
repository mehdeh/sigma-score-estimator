"""
Dataset loading and preprocessing for CIFAR-10.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset


def get_cifar10_dataloaders(
    batch_size=128,
    train_val_split=(0.85, 0.10),
    test_split=0.05,
    num_workers=4,
    data_root='./data'
):
    """
    Load CIFAR-10 dataset and create train/val/test dataloaders.
    
    Combines train and test sets, then splits into new train/val/test sets
    according to the specified ratios.
    
    Args:
        batch_size (int): Batch size for dataloaders
        train_val_split (tuple): (train_ratio, val_ratio) for splitting combined data
        test_split (float): Ratio of data to use for final test set
        num_workers (int): Number of worker processes for data loading
        data_root (str): Root directory for downloading/storing CIFAR-10 data
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load both train and test CIFAR-10 datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )

    # Combine the train and test datasets
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # Total size of the combined dataset
    total_size = len(combined_dataset)

    # Calculate the number of samples for each subset
    train_size = int(total_size * train_val_split[0])
    val_size = int(total_size * train_val_split[1])
    test_size = total_size - train_size - val_size

    # Split the combined dataset into train, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        combined_dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_cifar10_stats():
    """
    Get CIFAR-10 dataset statistics.
    
    Returns:
        dict: Statistics including image dimensions, number of classes, etc.
    """
    return {
        'num_classes': 10,
        'image_shape': (3, 32, 32),
        'image_dim': 3 * 32 * 32,  # 3072
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
    }


