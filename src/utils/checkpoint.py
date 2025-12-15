"""
Checkpoint utilities for saving and loading models.
Supports both .pkl (full model) and .pth (state dict) formats.
"""

import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_losses,
    val_losses,
    path,
    save_format='pkl'
):
    """
    Save a training checkpoint.
    
    Args:
        model (nn.Module): The model to save
        optimizer (Optimizer): The optimizer to save
        epoch (int): Current epoch number
        train_losses (list): List of training losses
        val_losses (list): List of validation losses
        path (str): Path to save the checkpoint
        save_format (str): Format to save ('pkl' or 'pth')
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if save_format == 'pkl':
        # Save entire model (self-contained, no architecture needed for loading)
        checkpoint = {
            'model': model,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, path)
    elif save_format == 'pth':
        # Save state dict only
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }
        torch.save(checkpoint, path)
    else:
        raise ValueError(f"Unknown save format: {save_format}. Expected 'pkl' or 'pth'.")
    
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model=None, optimizer=None, device='cuda'):
    """
    Load a checkpoint from file.
    
    Auto-detects format based on checkpoint contents.
    
    Args:
        path (str): Path to the checkpoint file
        model (nn.Module, optional): Model to load state dict into (for .pth format)
        optimizer (Optimizer, optional): Optimizer to load state dict into
        device (str or torch.device): Device to load checkpoint on
    
    Returns:
        dict: Dictionary containing loaded checkpoint data with keys:
            - 'model': The loaded model (for .pkl) or input model with loaded state (for .pth)
            - 'optimizer': Optimizer with loaded state if provided
            - 'epoch': Last epoch number
            - 'train_losses': List of training losses
            - 'val_losses': List of validation losses
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If .pth format is used but no model is provided
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Determine format and load accordingly
    if 'model' in checkpoint:
        # .pkl format - full model saved
        loaded_model = checkpoint['model']
        loaded_model = loaded_model.to(device)
        print(f"Loaded full model from {path}")
    elif 'model_state_dict' in checkpoint:
        # .pth format - state dict only
        if model is None:
            raise ValueError(
                "For .pth format, a model instance must be provided to load state dict into."
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model = model.to(device)
        print(f"Loaded model state dict from {path}")
    else:
        raise ValueError(f"Invalid checkpoint format in {path}")
    
    # Load optimizer state if optimizer is provided
    loaded_optimizer = None
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loaded_optimizer = optimizer
        print(f"Loaded optimizer state")
    
    # Extract training history
    epoch = checkpoint.get('epoch', 0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    
    return {
        'model': loaded_model,
        'optimizer': loaded_optimizer,
        'epoch': epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


def save_model_only(model, path, save_format='pkl'):
    """
    Save only the model (no optimizer or training history).
    
    Useful for exporting trained models for inference.
    
    Args:
        model (nn.Module): The model to save
        path (str): Path to save the model
        save_format (str): Format to save ('pkl' or 'pth')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if save_format == 'pkl':
        # Save entire model
        torch.save(model, path)
        print(f"Model saved to {path} (full model)")
    elif save_format == 'pth':
        # Save state dict only
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path} (state dict)")
    else:
        raise ValueError(f"Unknown save format: {save_format}. Expected 'pkl' or 'pth'.")


def load_model_only(path, model=None, device='cuda'):
    """
    Load only the model (no optimizer or training history).
    
    Args:
        path (str): Path to the model file
        model (nn.Module, optional): Model to load state dict into (for .pth format)
        device (str or torch.device): Device to load model on
    
    Returns:
        nn.Module: The loaded model
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Try to load as full model first
    try:
        loaded_model = torch.load(path, map_location=device)
        if isinstance(loaded_model, torch.nn.Module):
            print(f"Loaded full model from {path}")
            return loaded_model
    except:
        pass
    
    # If that fails, try to load as state dict
    if model is None:
        raise ValueError(
            "Could not load as full model. Please provide a model instance to load state dict into."
        )
    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Loaded model state dict from {path}")
    
    return model


def get_checkpoint_info(path):
    """
    Get information about a checkpoint without fully loading the model.
    
    Args:
        path (str): Path to the checkpoint file
    
    Returns:
        dict: Information about the checkpoint
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    
    checkpoint = torch.load(path, map_location='cpu')
    
    info = {
        'format': 'pkl' if 'model' in checkpoint else 'pth',
        'epoch': checkpoint.get('epoch', 'N/A'),
        'has_optimizer': 'optimizer_state_dict' in checkpoint,
        'train_losses_length': len(checkpoint.get('train_losses', [])),
        'val_losses_length': len(checkpoint.get('val_losses', [])),
    }
    
    if info['train_losses_length'] > 0:
        info['final_train_loss'] = checkpoint['train_losses'][-1]
    if info['val_losses_length'] > 0:
        info['final_val_loss'] = checkpoint['val_losses'][-1]
        info['best_val_loss'] = min(checkpoint['val_losses'])
    
    return info

