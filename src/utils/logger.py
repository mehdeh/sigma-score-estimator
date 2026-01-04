"""
Logging utilities for training and evaluation.
"""

import os
import json
import logging
import math
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name (str): Logger name
        log_file (str, optional): Path to log file
        level: Logging level
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """
    Logger for training/validation metrics.
    
    Saves metrics to JSON file for easy analysis and plotting.
    """
    
    def __init__(self, log_dir):
        """
        Initialize metrics logger.
        
        Args:
            log_dir (str): Directory to save metrics
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(log_dir, 'metrics.json')
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'epochs': [],
            'learning_rates': [],
            'timestamps': [],
        }
        
        # Load existing metrics if file exists
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
            except:
                pass
    
    def log_epoch(self, epoch, train_loss, val_loss, learning_rate=None):
        """
        Log metrics for an epoch.
        
        Args:
            epoch (int): Epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            learning_rate (float, optional): Current learning rate
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['train_losses'].append(float(train_loss))
        self.metrics['val_losses'].append(float(val_loss))
        self.metrics['timestamps'].append(datetime.now().isoformat())
        
        if learning_rate is not None:
            self.metrics['learning_rates'].append(float(learning_rate))
        
        # Save to file
        self.save()
    
    def save(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_best_epoch(self):
        """
        Get the epoch with the best (lowest) validation loss.
        
        Returns:
            tuple: (best_epoch, best_val_loss)
        """
        if not self.metrics['val_losses']:
            return None, None
        
        best_idx = self.metrics['val_losses'].index(min(self.metrics['val_losses']))
        best_epoch = self.metrics['epochs'][best_idx]
        best_val_loss = self.metrics['val_losses'][best_idx]
        
        return best_epoch, best_val_loss


def create_experiment_dir(base_dir='experiments', prefix='exp', config=None):
    """
    Create a new experiment directory with timestamp and important hyperparameters.
    
    Format: {prefix}_{timestamp}_{model}_{loss}_{epochs}e_{lr}_{bs}_{optimizer}
    Example: exp_20250104_060827_xsigma_chizscore_100e_lr1e-3_bs256_adam
    
    Args:
        base_dir (str): Base directory for experiments
        prefix (str): Prefix for experiment directory name
        config (dict, optional): Configuration dictionary containing hyperparameters
    
    Returns:
        str: Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name_parts = [prefix, timestamp]
    
    if config:
        # Add model type
        model_type = config.get('model', {}).get('type', 'unknown')
        # Shorten model type names for readability
        model_short = model_type.replace('omega_', '').replace('_', '')
        name_parts.append(model_short)
        
        # Add loss type (for train) or method (for test)
        training_config = config.get('training', {})
        loss_type = training_config.get('loss_type', '')
        if loss_type:
            # Shorten loss type names
            loss_short = loss_type.replace('omega_', '').replace('_', '')
            name_parts.append(loss_short)
        
        # Add number of epochs (for train)
        epochs = training_config.get('epochs')
        if epochs:
            name_parts.append(f'{epochs}e')
        
        # Add learning rate
        lr = training_config.get('learning_rate')
        if lr:
            # Format learning rate: 0.001 -> lr1e-3, 0.0001 -> lr1e-4, 0.01 -> lr1e-2
            if lr >= 1.0:
                lr_str = f'lr{int(lr)}'
            else:
                # For values < 1.0, use scientific notation
                lr_exp = int(abs(math.log10(lr)))
                lr_str = f'lr1e-{lr_exp}'
            name_parts.append(lr_str)
        
        # Add batch size
        batch_size = config.get('data', {}).get('batch_size')
        if batch_size:
            name_parts.append(f'bs{batch_size}')
        
        # Add optimizer type
        optimizer = training_config.get('optimizer', '')
        if optimizer:
            name_parts.append(optimizer)
    
    dir_name = '_'.join(name_parts)
    exp_dir = os.path.join(base_dir, dir_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir

