"""
Logging utilities for training and evaluation.
"""

import os
import json
import logging
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


def create_experiment_dir(base_dir='experiment', prefix='exp'):
    """
    Create a new experiment directory with timestamp.
    
    Args:
        base_dir (str): Base directory for experiments
        prefix (str): Prefix for experiment directory name
    
    Returns:
        str: Path to created experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'{prefix}_{timestamp}')
    
    # Create subdirectories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    
    return exp_dir

