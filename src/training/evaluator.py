"""
Evaluator class for evaluating trained omega estimator models.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..data import NoiseGenerator
from ..utils import (
    setup_logger,
    plot_predictions_scatter,
    plot_noise_distribution,
    visualize_sample_images,
)
from .loss_functions import LossFactory
from .output_transforms import TransformFactory


class OmegaEvaluator:
    """
    Evaluator for omega estimator models.
    
    Performs evaluation on test data and generates visualizations.
    
    Args:
        model (nn.Module): The trained model to evaluate
        test_loader (DataLoader): Test data loader
        config (dict): Configuration dictionary
        exp_dir (str): Experiment directory for saving outputs
        device (str or torch.device): Device to evaluate on
    """
    
    def __init__(
        self,
        model,
        test_loader,
        config,
        exp_dir,
        device='cuda'
    ):
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        self.test_loader = test_loader
        self.config = config
        self.exp_dir = exp_dir
        self.device = device
        
        # Extract config parameters
        self.model_type = config['model']['type']
        
        # Initialize noise generator
        self.noise_generator = NoiseGenerator(
            sigma_min=config['noise']['sigma_min'],
            sigma_max=config['noise']['sigma_max'],
            strategy=config['noise']['strategy'],
            num_samples=config['noise'].get('num_samples', 100),
            device=device
        )
        
        # Initialize loss function
        image_dim = 3 * 32 * 32  # CIFAR-10
        sigma_cal = config['training'].get('sigma_cal', 0.0)
        self.loss_fn = LossFactory.get_loss(
            config['training']['loss_type'],
            image_dim=image_dim,
            sigma_cal=sigma_cal
        )
        self.image_dim = image_dim
        
        # Initialize output transform
        self.output_transform = TransformFactory.get_transform(
            loss_type=config['training']['loss_type'],
            image_dim=image_dim,
            sigma_cal=sigma_cal
        )
        
        # Initialize logger
        log_file = os.path.join(exp_dir, 'test.log')
        self.logger = setup_logger('OmegaEvaluator', log_file)
        
        self.logger.info(f"Initialized OmegaEvaluator with model type: {self.model_type}")
        self.logger.info(f"Output transform: {type(self.output_transform).__name__}")
    
    def evaluate(self, num_samples=None, visualize=True):
        """
        Evaluate the model on test data.
        
        Args:
            num_samples (int, optional): Number of samples to evaluate (None for all)
            visualize (bool): Whether to generate visualizations
        
        Returns:
            dict: Evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        
        all_predictions = []
        all_targets = []
        all_sigmas = []
        all_losses = []
        
        # For visualization
        sample_clean_images = []
        sample_noisy_images = []
        sample_sigmas = []
        
        samples_evaluated = 0
        max_vis_samples = 10
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            
            for images, _ in pbar:
                if num_samples is not None and samples_evaluated >= num_samples:
                    break
                
                images = images.to(self.device)
                
                # Generate noise levels
                sigma = self.noise_generator.generate_sigma(images.size(0))
                
                # Add noise to images
                noisy_images, noise = self.noise_generator.add_noise(images, sigma)
                
                # Forward pass
                if self.model_type == 'omega_x':
                    output = self.model(noisy_images)
                else:  # omega_x_sigma
                    output = self.model(noisy_images, sigma)
                
                # Apply output transformation to get omega_hat
                # This converts model output to omega_hat for evaluation
                omega_hat = self.output_transform.apply(output, sigma=sigma)
                
                # Compute target (ground truth omega_hat)
                # All evaluation is done in omega_hat space
                target = self._compute_omega_hat_target(images, noisy_images, sigma)
                
                # Compute loss (for logging purposes)
                loss = self.loss_fn(output, images, noisy_images, sigma)
                
                # Store results (omega_hat predictions and targets)
                all_predictions.append(omega_hat.squeeze().cpu())
                all_targets.append(target.cpu())
                all_sigmas.append(sigma.cpu())
                all_losses.append(loss.item())
                
                # Store samples for visualization
                if len(sample_clean_images) < max_vis_samples:
                    n_to_add = min(max_vis_samples - len(sample_clean_images), images.size(0))
                    sample_clean_images.append(images[:n_to_add].cpu())
                    sample_noisy_images.append(noisy_images[:n_to_add].cpu())
                    sample_sigmas.append(sigma[:n_to_add].cpu())
                
                samples_evaluated += images.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_sigmas = torch.cat(all_sigmas).numpy()
        
        # Compute metrics
        mse = mean_squared_error(all_targets, all_predictions)
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        avg_loss = np.mean(all_losses)
        
        # Compute relative error metrics
        relative_errors = np.abs(all_predictions - all_targets) / (np.abs(all_targets) + 1e-8)
        mean_relative_error = np.mean(relative_errors)
        median_relative_error = np.median(relative_errors)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2_score': float(r2),
            'avg_loss': float(avg_loss),
            'mean_relative_error': float(mean_relative_error),
            'median_relative_error': float(median_relative_error),
            'num_samples': samples_evaluated,
        }
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Number of samples: {samples_evaluated}")
        self.logger.info(f"  MSE: {mse:.6f}")
        self.logger.info(f"  MAE: {mae:.6f}")
        self.logger.info(f"  R² Score: {r2:.6f}")
        self.logger.info(f"  Average Loss: {avg_loss:.6f}")
        self.logger.info(f"  Mean Relative Error: {mean_relative_error:.4f}")
        self.logger.info(f"  Median Relative Error: {median_relative_error:.4f}")
        
        # Save metrics to JSON
        import json
        metrics_file = os.path.join(self.exp_dir, 'test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
        
        # Generate visualizations
        if visualize:
            self.logger.info("Generating visualizations...")
            plots_dir = self.exp_dir
            
            # Scatter plot of predictions vs targets
            scatter_path = os.path.join(plots_dir, 'test_scatter_predictions.png')
            plot_predictions_scatter(
                all_predictions,
                all_targets,
                save_path=scatter_path,
                show=False,
                title='Model Predictions vs Ground Truth'
            )
            
            # Noise distribution plot
            noise_dist_path = os.path.join(plots_dir, 'test_noise_distribution.png')
            plot_noise_distribution(
                all_sigmas,
                save_path=noise_dist_path,
                show=False
            )
            
            # Sample images visualization
            if sample_clean_images:
                sample_clean_images = torch.cat(sample_clean_images)
                sample_noisy_images = torch.cat(sample_noisy_images)
                sample_sigmas = torch.cat(sample_sigmas)
                
                sample_images_path = os.path.join(plots_dir, 'test_sample_images.png')
                visualize_sample_images(
                    sample_clean_images,
                    sample_noisy_images,
                    sample_sigmas,
                    n_samples=min(5, len(sample_clean_images)),
                    save_path=sample_images_path,
                    show=False
                )
            
            self.logger.info(f"Visualizations saved to {plots_dir}")
        
        self.logger.info("Evaluation completed!")
        
        return metrics
    
    def _compute_omega_hat_target(self, clean_images, noisy_images, sigma):
        """
        Compute ground truth omega_hat for evaluation.
        
        All evaluation metrics are computed in omega_hat space after transformation.
        This provides a consistent evaluation metric across all loss types.
        
        Args:
            clean_images (Tensor): Clean images, shape (batch_size, C, H, W)
            noisy_images (Tensor): Noisy images, shape (batch_size, C, H, W)
            sigma (Tensor): Noise levels, shape (batch_size,)
        
        Returns:
            Tensor: Ground truth omega_hat = ||epsilon||^2 / sigma
        """
        batch_size = clean_images.size(0)
        
        # Flatten images
        clean_flat = clean_images.view(batch_size, -1)
        noisy_flat = noisy_images.view(batch_size, -1)
        
        # Ensure sigma is 1D
        if sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Compute epsilon = (x - x_tilde) / sigma
        epsilon = (noisy_flat - clean_flat) / sigma.view(-1, 1)
        
        # Compute ||epsilon||^2
        epsilon_norm_sq = torch.sum(epsilon ** 2, dim=1)
        
        # Ground truth omega_hat = ||epsilon||^2 / sigma
        omega_hat_target = epsilon_norm_sq / sigma
        
        return omega_hat_target
    
    def predict_batch(self, images, sigma):
        """
        Make predictions for a batch of images.
        
        Applies the appropriate output transformation to convert model output to omega_hat.
        
        Args:
            images (Tensor): Clean images (batch_size, C, H, W)
            sigma (Tensor or float): Noise levels
        
        Returns:
            Tensor: Transformed predictions (omega_hat)
        """
        self.model.eval()
        
        images = images.to(self.device)
        
        # Convert sigma to tensor if needed
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.full((images.size(0),), sigma, device=self.device)
        else:
            sigma = sigma.to(self.device)
        
        # Add noise
        noisy_images, _ = self.noise_generator.add_noise(images, sigma)
        
        with torch.no_grad():
            if self.model_type == 'omega_x':
                output = self.model(noisy_images)
            else:  # omega_x_sigma
                output = self.model(noisy_images, sigma)
            
            # Apply output transformation
            omega_hat = self.output_transform.apply(output, sigma=sigma)
        
        return omega_hat

