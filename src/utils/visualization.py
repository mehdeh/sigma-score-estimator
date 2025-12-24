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
    
    For omega estimation, this plots:
    - X-axis (Target): ω̂_target = ||x - x̃||² / σ³ (ground truth)
    - Y-axis (Prediction): ω̂ from model output after transformation
    
    Args:
        predictions (array-like): Model predictions (ω̂)
        targets (array-like): Ground truth targets (ω̂_target)
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
    
    plt.xlabel('Target ω̂ = ||x - x̃||² / σ³', fontsize=12)
    plt.ylabel('Predicted ω̂ (Model Output)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add R^2 score and sample count
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    stats_text = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nN = {len(targets)}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
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


def plot_error_vs_sigma(predictions, targets, sigma_values, save_path=None, show=False,
                        title='Prediction Error vs Sigma', num_bins=20):
    """
    Plot the relationship between prediction error and sigma values.
    
    Creates a visualization showing how prediction error varies with noise level (sigma).
    This helps identify whether model performance degrades at certain noise levels.
    
    The plot includes:
    - Scatter plot of absolute errors vs sigma
    - Binned average errors with error bars (showing variation within each bin)
    
    Args:
        predictions (array-like): Model predictions (ω̂)
        targets (array-like): Ground truth targets (ω̂_target)
        sigma_values (array-like): Noise level values
        save_path (str, optional): Path to save the plot
        show (bool): Whether to display the plot
        title (str): Plot title
        num_bins (int): Number of bins for aggregating errors by sigma range
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if isinstance(sigma_values, torch.Tensor):
        sigma_values = sigma_values.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    sigma_values = sigma_values.flatten()
    
    # Compute absolute errors
    absolute_errors = np.abs(predictions - targets)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left subplot: Scatter plot of errors vs sigma
    ax1.scatter(sigma_values, absolute_errors, alpha=0.3, s=10, c='steelblue')
    ax1.set_xlabel('Sigma (Noise Level)', fontsize=12)
    ax1.set_ylabel('Absolute Error |Predicted ω̂ - Target ω̂|', fontsize=12)
    ax1.set_title(f'{title} - Scatter Plot', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add correlation info
    from scipy.stats import pearsonr, spearmanr
    try:
        pearson_corr, pearson_p = pearsonr(sigma_values, absolute_errors)
        spearman_corr, spearman_p = spearmanr(sigma_values, absolute_errors)
        corr_text = f'Pearson r = {pearson_corr:.3f} (p={pearson_p:.4f})\n'
        corr_text += f'Spearman ρ = {spearman_corr:.3f} (p={spearman_p:.4f})'
        ax1.text(0.05, 0.95, corr_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    except Exception as e:
        print(f"Warning: Could not compute correlation: {e}")
    
    # Right subplot: Binned average errors with error bars
    sigma_min, sigma_max = sigma_values.min(), sigma_values.max()
    bin_edges = np.linspace(sigma_min, sigma_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (sigma_values >= bin_edges[i]) & (sigma_values < bin_edges[i + 1])
        if i == num_bins - 1:  # Include the last edge in the last bin
            mask = (sigma_values >= bin_edges[i]) & (sigma_values <= bin_edges[i + 1])
        
        errors_in_bin = absolute_errors[mask]
        
        if len(errors_in_bin) > 0:
            bin_means.append(np.mean(errors_in_bin))
            bin_stds.append(np.std(errors_in_bin))
            bin_counts.append(len(errors_in_bin))
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_counts.append(0)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_counts = np.array(bin_counts)
    
    # Plot binned errors with error bars
    ax2.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', 
                linewidth=2, markersize=6, capsize=5, capthick=2,
                color='darkred', ecolor='coral', label='Mean ± Std')
    
    ax2.set_xlabel('Sigma (Noise Level)', fontsize=12)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title(f'{title} - Binned Statistics', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Add sample count info
    stats_text = f'Total samples: {len(sigma_values)}\n'
    stats_text += f'Bins: {num_bins}\n'
    stats_text += f'Mean error: {np.mean(absolute_errors):.4f}\n'
    stats_text += f'Std error: {np.std(absolute_errors):.4f}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error vs sigma plot saved to {save_path}")
    
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

