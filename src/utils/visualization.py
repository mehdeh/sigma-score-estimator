"""
Visualization utilities for training progress and model predictions.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_loss_curves(train_losses, val_losses, save_path=None, show=False):
    """
    Plot training and validation loss curves.
    
    Creates two subplots: regular loss and logarithmic loss.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    epochs = range(1, len(train_losses) + 1)
    
    # Convert to numpy arrays for easier manipulation
    train_losses_arr = np.array(train_losses)
    val_losses_arr = np.array(val_losses)
    
    # Left subplot: Regular loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add best epoch marker on left plot
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                    label=f'Best Epoch: {best_epoch}')
        ax1.plot(best_epoch, best_loss, 'g*', markersize=15)
        ax1.legend(fontsize=10)
    
    # Right subplot: Logarithmic loss
    ax2.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax2.set_title('Training and Validation Loss (Logarithmic)', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    # Add best epoch marker on right plot
    if val_losses:
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, 
                    label=f'Best Epoch: {best_epoch}')
        ax2.plot(best_epoch, best_loss, 'g*', markersize=15)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_predictions_scatter(predictions, targets, save_path=None, show=False, title='Predictions vs Targets'):
    """
    Create a scatter plot comparing predictions to targets.
    
    Args:
        predictions (array-like): Model predictions
        targets (array-like): Ground truth targets
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
        title (str): Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Create scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=10)
    
    # Add perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Target', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add R^2 score
    from sklearn.metrics import r2_score
    r2 = r2_score(targets, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_noise_distribution(sigma_values, save_path=None, show=False):
    """
    Plot histogram of noise level (sigma) distribution.
    
    Args:
        sigma_values (array-like): Noise level values
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Convert to numpy if needed
    if isinstance(sigma_values, torch.Tensor):
        sigma_values = sigma_values.detach().cpu().numpy()
    
    sigma_values = sigma_values.flatten()
    
    # Create histogram
    plt.hist(sigma_values, bins=50, alpha=0.7, edgecolor='black')
    
    plt.xlabel('Sigma (Noise Level)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Noise Levels', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_sigma = np.mean(sigma_values)
    std_sigma = np.std(sigma_values)
    plt.axvline(mean_sigma, color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_sigma:.3f}')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Noise distribution plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_sample_images(clean_images, noisy_images, sigma_values, n_samples=5, 
                            save_path=None, show=False):
    """
    Visualize clean and noisy image pairs.
    
    Args:
        clean_images (Tensor): Clean images (batch_size, C, H, W)
        noisy_images (Tensor): Noisy images (batch_size, C, H, W)
        sigma_values (Tensor): Noise levels (batch_size,)
        n_samples (int): Number of image pairs to visualize
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
    """
    n_samples = min(n_samples, clean_images.size(0))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
    
    # Move tensors to CPU and denormalize
    clean_images = clean_images.cpu()
    noisy_images = noisy_images.cpu()
    sigma_values = sigma_values.cpu()
    
    # Denormalize from [-1, 1] to [0, 1]
    clean_images = (clean_images + 1) / 2
    noisy_images = (noisy_images + 1) / 2
    
    # Clip to valid range
    clean_images = torch.clamp(clean_images, 0, 1)
    noisy_images = torch.clamp(noisy_images, 0, 1)
    
    for i in range(n_samples):
        # Clean image
        axes[0, i].imshow(clean_images[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title('Clean')
        axes[0, i].axis('off')
        
        # Noisy image
        axes[1, i].imshow(noisy_images[i].permute(1, 2, 0).numpy())
        axes[1, i].set_title(f'Noisy (σ={sigma_values[i]:.2f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample images saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

