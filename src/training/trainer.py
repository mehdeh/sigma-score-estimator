"""
Trainer class for training the omega estimator models.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from ..data import NoiseGenerator
from ..utils import (
    save_checkpoint,
    setup_logger,
    MetricsLogger,
    plot_loss_curves,
)
from .loss_functions import LossFactory


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
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training'].get('weight_decay', 0.0)
        self.early_stopping = config['training'].get('early_stopping', True)
        self.patience = config['training'].get('patience', 10)
        self.log_interval = config['logging'].get('log_interval', 100)
        self.save_format = config['checkpoint'].get('save_format', 'pkl')
        self.save_every = config['checkpoint'].get('save_every', 10)
        
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
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Initialize learning rate scheduler
        scheduler_type = config['training'].get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['scheduler_params'].get('T_max', self.epochs)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['scheduler_params'].get('step_size', 30),
                gamma=config['training']['scheduler_params'].get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        # Initialize logger
        log_file = os.path.join(exp_dir, 'train.log')
        self.logger = setup_logger('OmegaTrainer', log_file)
        
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
            
            # Save checkpoint periodically
            if (epoch + 1) % self.save_every == 0:
                checkpoint_path = os.path.join(
                    self.exp_dir, f'checkpoint_epoch_{epoch+1}.{self.save_format}'
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.train_losses,
                    self.val_losses,
                    checkpoint_path,
                    save_format=self.save_format
                )
            
            # Save best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                best_path = os.path.join(
                    self.exp_dir, f'best_model.{self.save_format}'
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
        
        # Save final checkpoint
        final_path = os.path.join(
            self.exp_dir, f'latest.{self.save_format}'
        )
        save_checkpoint(
            self.model,
            self.optimizer,
            epoch,
            self.train_losses,
            self.val_losses,
            final_path,
            save_format=self.save_format
        )
        
        self.logger.info("Training completed!")
        best_epoch, best_val_loss = self.metrics_logger.get_best_epoch()
        self.logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
        }

