"""
Trainer class for training the omega estimator models.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..datasets import NoiseGenerator
from ..utils import (
    save_checkpoint,
    setup_logger,
    MetricsLogger,
    plot_loss_curves,
    plot_predictions_scatter,
    plot_error_vs_sigma,
)
from .loss_functions import LossFactory
from .output_transforms import TransformFactory


class OmegaTrainer:
    """
    Trainer for omega estimator models.
    
    Handles training and validation loops, checkpointing, and logging.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (dict): Configuration dictionary
        exp_dir (str): Experiment directory for saving outputs
        device (str or torch.device): Device to train on
        resume_checkpoint (str, optional): Path to checkpoint to resume from
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        exp_dir,
        device='cuda',
        resume_checkpoint=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.exp_dir = exp_dir
        self.device = device
        
        # Extract config parameters
        self.model_type = config['model']['type']
        self.loss_type = config['training']['loss_type']
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 0.0)
        self.early_stopping = config['training'].get('early_stopping', True)
        self.patience = config['training'].get('patience', 10)
        self.log_interval = config['logging'].get('log_interval', 100)
        self.save_format = config['checkpoint'].get('save_format', 'pkl')
        
        # Initialize noise generator
        self.noise_generator = NoiseGenerator(
            sigma_min=config['noise']['sigma_min'],
            sigma_max=config['noise']['sigma_max'],
            strategy=config['noise']['strategy'],
            num_samples=config['noise'].get('num_samples', 100),
            device=device
        )
        
        # Initialize logger (must be before optimizer/scheduler creation)
        log_file = os.path.join(exp_dir, 'train.log')
        self.logger = setup_logger('OmegaTrainer', log_file)
        
        # Initialize loss function
        image_dim = 3 * 32 * 32  # CIFAR-10
        self.loss_fn = LossFactory.get_loss(
            self.loss_type,
            image_dim=image_dim
        )
        self.image_dim = image_dim
        
        # Initialize output transform for evaluation
        self.output_transform = TransformFactory.get_transform(
            loss_type=self.loss_type,
            image_dim=image_dim
        )
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(config)
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler(config)
        
        # Initialize metrics logger
        self.metrics_logger = MetricsLogger(exp_dir)
        
        # Training state
        self.start_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Resume from checkpoint if provided
        if resume_checkpoint is not None:
            self._resume_from_checkpoint(resume_checkpoint)
        
        self.logger.info(f"Initialized OmegaTrainer with model type: {self.model_type}")
        self.logger.info(f"Loss function: {config['training']['loss_type']}")
        self.logger.info(f"Noise strategy: {config['noise']['strategy']}")
        self.logger.info(f"Optimizer: {config['training'].get('optimizer', 'adam').upper()}")
        if self.scheduler is not None:
            self.logger.info(f"LR Scheduler: {config['training'].get('scheduler', 'none')}")
    
    def _create_optimizer(self, config):
        """
        Create optimizer based on config.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer_type = config['training'].get('optimizer', 'adam').lower()
        optimizer_params = config['training'].get('optimizer_params', {})
        
        if optimizer_type == 'adam':
            betas = optimizer_params.get('betas', [0.9, 0.999])
            eps = optimizer_params.get('eps', 1e-8)
            
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=tuple(betas),
                eps=eps,
                weight_decay=self.weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = optimizer_params.get('momentum', 0.9)
            nesterov = optimizer_params.get('nesterov', True)
            
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self, config):
        """
        Create learning rate scheduler based on config.
        
        Args:
            config (dict): Configuration dictionary
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler or None: Configured scheduler
        """
        scheduler_type = config['training'].get('scheduler', 'cosine').lower()
        scheduler_params = config['training'].get('scheduler_params', {})
        
        if scheduler_type == 'cosine':
            # T_max should be >= total epochs to avoid LR cycling
            # If T_max is None or not set, use epochs value
            t_max = scheduler_params.get('T_max')
            if t_max is None:
                t_max = self.epochs
                self.logger.info(f"T_max not set, using epochs value: {t_max}")
            
            if t_max < self.epochs:
                self.logger.warning(
                    f"T_max ({t_max}) is less than epochs ({self.epochs}). "
                    f"This will cause LR to cycle and may degrade performance."
                )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max
            )
        elif scheduler_type == 'step':
            step_size = scheduler_params.get('step_size', 30)
            gamma = scheduler_params.get('gamma', 0.1)
            
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
        
        return scheduler
    
    def _resume_from_checkpoint(self, checkpoint_path):
        """Resume training from a checkpoint."""
        from ..utils import load_checkpoint
        
        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )
        
        self.model = checkpoint['model']
        if checkpoint['optimizer'] is not None:
            self.optimizer = checkpoint['optimizer']
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        if self.val_losses:
            self.best_val_loss = min(self.val_losses)
        
        self.logger.info(f"Resumed training from epoch {self.start_epoch}")
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.start_epoch + self.epochs}",
            leave=False
        )
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            
            # Generate noise levels
            sigma = self.noise_generator.generate_sigma(images.size(0))
            
            # Add noise to images
            noisy_images, noise = self.noise_generator.add_noise(images, sigma)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'omega_x':
                output = self.model(noisy_images)
            else:  # omega_x_sigma
                output = self.model(noisy_images, sigma)
            
            # Compute loss
            loss = self.loss_fn(output, images, noisy_images, sigma)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log at intervals
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = running_loss / num_batches
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.start_epoch + self.epochs}] "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f})"
                )
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, epoch):
        """
        Validate for one epoch.
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            float: Average validation loss for the epoch
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Validation",
                leave=False
            )
            
            for images, _ in pbar:
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
                
                # Compute loss
                loss = self.loss_fn(output, images, noisy_images, sigma)
                
                # Update statistics
                running_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = running_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            dict: Training results containing final losses and best epoch
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.epochs} epochs")
        self.logger.info(f"Model type: {self.model_type}")
        
        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            self.val_losses.append(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log epoch results
            self.logger.info(
                f"Epoch [{epoch+1}/{self.start_epoch + self.epochs}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}"
            )
            
            # Log metrics
            self.metrics_logger.log_epoch(epoch + 1, train_loss, val_loss, current_lr)
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                best_path = os.path.join(
                    self.exp_dir, f'checkpoint_best_model.{self.save_format}'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.train_losses,
                    self.val_losses,
                    best_path,
                    save_format=self.save_format
                )
                self.logger.info(f"New best model saved with val loss: {val_loss:.6f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if self.early_stopping and self.epochs_without_improvement >= self.patience:
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs without improvement"
                )
                break
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Plot loss curves
            plot_path = os.path.join(self.exp_dir, 'loss_curve.png')
            plot_loss_curves(
                self.train_losses,
                self.val_losses,
                save_path=plot_path,
                show=False
            )
            
            # Save latest checkpoint after each epoch
            latest_path = os.path.join(
                self.exp_dir, f'checkpoint_latest.{self.save_format}'
            )
            save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                self.train_losses,
                self.val_losses,
                latest_path,
                save_format=self.save_format
            )
        
        self.logger.info("Training completed!")
        best_epoch, best_val_loss = self.metrics_logger.get_best_epoch()
        self.logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        
        # Generate scatter plots and compute metrics for final training evaluation
        self.logger.info("Generating training evaluation plots and computing metrics...")
        self._generate_evaluation_plots_and_metrics()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }
    
    def _compute_omega_hat_target(self, clean_images, noisy_images, sigma):
        """
        Compute ground truth omega_hat for evaluation.
        
        Computes: ω̂_target = ||x - x̃||² / σ³
        
        This is mathematically equivalent to: ||ε||² / σ
        where ε = (x̃ - x) / σ
        
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
    
    def _compute_raw_target(self, clean_images, noisy_images, sigma):
        """
        Compute ground truth in the same space as raw model output (before transformation).
        
        For omega_chi_zscore: Computes z-score = (||ε||² - d) / sqrt(2*d)
        For other losses: Same as omega_hat target (no transformation)
        
        Args:
            clean_images (Tensor): Clean images, shape (batch_size, C, H, W)
            noisy_images (Tensor): Noisy images, shape (batch_size, C, H, W)
            sigma (Tensor): Noise levels, shape (batch_size,)
        
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
        if self.loss_type == 'omega_chi_zscore':
            # z-score = (||epsilon||² - d) / sqrt(2*d)
            sqrt_2d = torch.sqrt(torch.tensor(2.0 * d, device=epsilon_norm_sq.device))
            raw_target = (epsilon_norm_sq - d) / sqrt_2d
        else:
            # For other losses, raw output is already omega_hat
            raw_target = epsilon_norm_sq / sigma
        
        return raw_target
    
    def _evaluate_dataset(self, data_loader, max_batches=None):
        """
        Evaluate model on a dataset and collect predictions vs targets.
        
        Args:
            data_loader (DataLoader): Data loader to evaluate on
            max_batches (int, optional): Maximum number of batches to evaluate
        
        Returns:
            tuple: (predictions, targets, sigmas, raw_predictions, raw_targets, losses) as numpy arrays
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_sigmas = []
        all_raw_predictions = []
        all_raw_targets = []
        all_losses = []
        
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                images = images.to(self.device)
                
                # Generate noise levels
                sigma = self.noise_generator.generate_sigma(images.size(0))
                
                # Add noise to images
                noisy_images, _ = self.noise_generator.add_noise(images, sigma)
                
                # Forward pass
                if self.model_type == 'omega_x':
                    output = self.model(noisy_images)
                else:  # omega_x_sigma
                    output = self.model(noisy_images, sigma)
                
                # Store raw output (before transformation)
                all_raw_predictions.append(output.squeeze().cpu())
                
                # Apply output transformation to get omega_hat
                omega_hat = self.output_transform.apply(output, sigma=sigma)
                
                # Compute target omega_hat using the correct formula
                target = self._compute_omega_hat_target(images, noisy_images, sigma)
                
                # Compute raw target (in same space as raw model output)
                raw_target = self._compute_raw_target(images, noisy_images, sigma)
                
                # Compute loss
                loss = self.loss_fn(output, images, noisy_images, sigma)
                
                # Store results
                all_predictions.append(omega_hat.squeeze().cpu())
                all_targets.append(target.cpu())
                all_sigmas.append(sigma.cpu())
                all_raw_targets.append(raw_target.cpu())
                all_losses.append(loss.item())
        
        # Concatenate and convert to numpy
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()
        sigmas = torch.cat(all_sigmas).numpy()
        raw_predictions = torch.cat(all_raw_predictions).numpy()
        raw_targets = torch.cat(all_raw_targets).numpy()
        losses = np.array(all_losses)
        
        return predictions, targets, sigmas, raw_predictions, raw_targets, losses
    
    def _compute_metrics(self, predictions, targets, losses):
        """
        Compute evaluation metrics from predictions and targets.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): Ground truth targets
            losses (np.ndarray): Loss values for each sample
        
        Returns:
            dict: Dictionary containing evaluation metrics
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
    
    def _save_metrics(self, metrics, filename):
        """
        Save metrics to JSON file.
        
        Args:
            metrics (dict): Dictionary of metrics to save
            filename (str): Name of the metrics file (e.g., 'train_metrics.json')
        """
        metrics_file = os.path.join(self.exp_dir, filename)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
    
    def _generate_evaluation_plots_and_metrics(self):
        """
        Generate evaluation plots and compute metrics for training and validation data.
        
        Creates multiple plots:
        1. Scatter plots showing model predictions vs ground truth omega_hat
        2. Error vs sigma plots showing how prediction error varies with noise level
        
        Also computes and saves metrics to JSON files similar to test evaluation.
        """
        try:
            # Evaluate on training data (use subset to save time)
            self.logger.info("Evaluating on training data...")
            train_predictions, train_targets, train_sigmas, train_raw_predictions, train_raw_targets, train_losses = self._evaluate_dataset(
                self.train_loader, 
                max_batches=50  # Evaluate on first 50 batches
            )
            
            # Compute training metrics
            train_metrics = self._compute_metrics(train_predictions, train_targets, train_losses)
            
            # Log training metrics
            self.logger.info("Training Evaluation Results:")
            self.logger.info(f"  Number of samples: {train_metrics['num_samples']}")
            self.logger.info(f"  MSE: {train_metrics['mse']:.6f}")
            self.logger.info(f"  MAE: {train_metrics['mae']:.6f}")
            self.logger.info(f"  R² Score: {train_metrics['r2_score']:.6f}")
            self.logger.info(f"  Average Loss: {train_metrics['avg_loss']:.6f}")
            self.logger.info(f"  Mean Relative Error: {train_metrics['mean_relative_error']:.4f}")
            self.logger.info(f"  Median Relative Error: {train_metrics['median_relative_error']:.4f}")
            
            # Save training metrics
            self._save_metrics(train_metrics, 'train_metrics.json')
            
            # Generate training scatter plot (showing both before and after transformation)
            train_scatter_path = os.path.join(self.exp_dir, 'train_scatter_predictions.png')
            plot_predictions_scatter(
                train_predictions,
                train_targets,
                save_path=train_scatter_path,
                show=False,
                title='Training: Model Predictions vs Ground Truth ω̂',
                raw_predictions=train_raw_predictions,
                raw_targets=train_raw_targets,
                loss_type=self.loss_type
            )
            self.logger.info(f"Training scatter plot saved to {train_scatter_path}")
            
            # Generate training error vs sigma plot
            train_error_sigma_path = os.path.join(self.exp_dir, 'train_error_vs_sigma.png')
            plot_error_vs_sigma(
                train_predictions,
                train_targets,
                train_sigmas,
                save_path=train_error_sigma_path,
                show=False,
                title='Training: Prediction Error vs Sigma'
            )
            self.logger.info(f"Training error vs sigma plot saved to {train_error_sigma_path}")
            
            # Evaluate on validation data
            self.logger.info("Evaluating on validation data...")
            val_predictions, val_targets, val_sigmas, val_raw_predictions, val_raw_targets, val_losses = self._evaluate_dataset(
                self.val_loader,
                max_batches=None  # Evaluate on full validation set
            )
            
            # Compute validation metrics
            val_metrics = self._compute_metrics(val_predictions, val_targets, val_losses)
            
            # Log validation metrics
            self.logger.info("Validation Evaluation Results:")
            self.logger.info(f"  Number of samples: {val_metrics['num_samples']}")
            self.logger.info(f"  MSE: {val_metrics['mse']:.6f}")
            self.logger.info(f"  MAE: {val_metrics['mae']:.6f}")
            self.logger.info(f"  R² Score: {val_metrics['r2_score']:.6f}")
            self.logger.info(f"  Average Loss: {val_metrics['avg_loss']:.6f}")
            self.logger.info(f"  Mean Relative Error: {val_metrics['mean_relative_error']:.4f}")
            self.logger.info(f"  Median Relative Error: {val_metrics['median_relative_error']:.4f}")
            
            # Save validation metrics
            self._save_metrics(val_metrics, 'val_metrics.json')
            
            # Generate validation scatter plot (showing both before and after transformation)
            val_scatter_path = os.path.join(self.exp_dir, 'val_scatter_predictions.png')
            plot_predictions_scatter(
                val_predictions,
                val_targets,
                save_path=val_scatter_path,
                show=False,
                title='Validation: Model Predictions vs Ground Truth ω̂',
                raw_predictions=val_raw_predictions,
                raw_targets=val_raw_targets,
                loss_type=self.loss_type
            )
            self.logger.info(f"Validation scatter plot saved to {val_scatter_path}")
            
            # Generate validation error vs sigma plot
            val_error_sigma_path = os.path.join(self.exp_dir, 'val_error_vs_sigma.png')
            plot_error_vs_sigma(
                val_predictions,
                val_targets,
                val_sigmas,
                save_path=val_error_sigma_path,
                show=False,
                title='Validation: Prediction Error vs Sigma'
            )
            self.logger.info(f"Validation error vs sigma plot saved to {val_error_sigma_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation plots and metrics: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

