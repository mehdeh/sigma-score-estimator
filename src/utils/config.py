"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to YAML config file
    
    Returns:
        dict: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Config saved to {save_path}")


def merge_configs(base_config, override_config):
    """
    Merge two configuration dictionaries.
    
    Values in override_config take precedence over base_config.
    
    Args:
        base_config (dict): Base configuration
        override_config (dict): Override configuration
    
    Returns:
        dict: Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config):
    """
    Validate configuration dictionary.
    
    Checks for required fields and valid values.
    
    Args:
        config (dict): Configuration to validate
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Required top-level keys
    required_keys = ['model', 'noise', 'training', 'data']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    # Validate model config
    if 'type' not in config['model']:
        raise ValueError("Model config must specify 'type'")
    
    valid_model_types = ['omega_x', 'omega_x_sigma']
    if config['model']['type'] not in valid_model_types:
        raise ValueError(
            f"Invalid model type: {config['model']['type']}. "
            f"Must be one of {valid_model_types}"
        )
    
    # Validate noise config
    if 'sigma_min' not in config['noise'] or 'sigma_max' not in config['noise']:
        raise ValueError("Noise config must specify 'sigma_min' and 'sigma_max'")
    
    if config['noise']['sigma_min'] >= config['noise']['sigma_max']:
        raise ValueError("sigma_min must be less than sigma_max")
    
    valid_noise_strategies = ['uniform', 'log_uniform', 'select_batch']
    if config['noise'].get('strategy', 'uniform') not in valid_noise_strategies:
        raise ValueError(
            f"Invalid noise strategy. Must be one of {valid_noise_strategies}"
        )
    
    # Validate training config
    if 'loss_type' not in config['training']:
        raise ValueError("Training config must specify 'loss_type'")
    
    valid_loss_types = [
        'omega_hat', 'omega_epsilon', 'omega_chi_zscore'
    ]
    if config['training']['loss_type'] not in valid_loss_types:
        raise ValueError(
            f"Invalid loss type: {config['training']['loss_type']}. "
            f"Must be one of {valid_loss_types}"
        )
    
    # Validate data config
    if 'batch_size' not in config['data']:
        raise ValueError("Data config must specify 'batch_size'")
    
    print("Configuration validated successfully")


def create_default_config():
    """
    Create a default configuration dictionary.
    
    Returns:
        dict: Default configuration
    """
    return {
        'model': {
            'type': 'omega_x_sigma',
            'backbone': 'resnet18',
        },
        'noise': {
            'sigma_min': 0.01,
            'sigma_max': 10.0,
            'strategy': 'uniform',
            'num_samples': 100,
        },
        'training': {
            'loss_type': 'omega_hat',
            'epochs': 100,
            'optimizer': 'adam',
            'optimizer_params': {
                'betas': [0.9, 0.999],
                'eps': 1.0e-8,
                'momentum': 0.9,
                'nesterov': True,
            },
            'learning_rate': 0.001,
            'weight_decay': 0.0,
            'scheduler': 'cosine',
            'scheduler_params': {
                'T_max': None,
                'step_size': 30,
                'gamma': 0.1,
            },
            'early_stopping': True,
            'patience': 10,
        },
        'data': {
            'batch_size': 128,
            'num_workers': 4,
            'train_val_split': [0.85, 0.10],
            'test_split': 0.05,
            'data_root': './data',
            'augmentation': {
                'enabled': True,
                'horizontal_flip': True,
                'vertical_flip': True,
                'rotation': True,
                'gaussian_noise': True,
                'gaussian_noise_std': 0.05,
                'flip_prob': 0.5,
                'rotation_prob': 0.5,
                'noise_prob': 0.3,
            },
        },
        'checkpoint': {
            'save_format': 'pkl',
            'save_best': True,
        },
        'logging': {
            'log_interval': 100,
            'plot_interval': 1,
        },
        'device': 'cuda',
        'seed': 42,
    }


def override_config_with_args(config, args):
    """
    Override configuration with command-line arguments.
    
    Args:
        config (dict): Base configuration
        args (argparse.Namespace): Command-line arguments
    
    Returns:
        dict: Updated configuration
    """
    # Model overrides
    if hasattr(args, 'model_type') and args.model_type is not None:
        config['model']['type'] = args.model_type
    
    # Noise overrides
    if hasattr(args, 'sigma_min') and args.sigma_min is not None:
        config['noise']['sigma_min'] = args.sigma_min
    if hasattr(args, 'sigma_max') and args.sigma_max is not None:
        config['noise']['sigma_max'] = args.sigma_max
    if hasattr(args, 'noise_strategy') and args.noise_strategy is not None:
        config['noise']['strategy'] = args.noise_strategy
    
    # Model overrides (for computational methods)
    if hasattr(args, 'sigma_threshold') and args.sigma_threshold is not None:
        if 'model' not in config:
            config['model'] = {}
        config['model']['sigma_threshold'] = args.sigma_threshold
    
    # Training overrides
    if hasattr(args, 'loss_type') and args.loss_type is not None:
        config['training']['loss_type'] = args.loss_type
    if hasattr(args, 'epochs') and args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if hasattr(args, 'optimizer') and args.optimizer is not None:
        config['training']['optimizer'] = args.optimizer
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
    if hasattr(args, 'scheduler') and args.scheduler is not None:
        config['training']['scheduler'] = args.scheduler
    if hasattr(args, 'early_stopping') and args.early_stopping is not None:
        config['training']['early_stopping'] = args.early_stopping
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    
    # Device override
    if hasattr(args, 'device') and args.device is not None:
        config['device'] = args.device
    
    # Seed override
    if hasattr(args, 'seed') and args.seed is not None:
        config['seed'] = args.seed
    
    # Data augmentation overrides
    if hasattr(args, 'augmentation') and args.augmentation is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['enabled'] = args.augmentation
    
    if hasattr(args, 'aug_hflip') and args.aug_hflip is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['horizontal_flip'] = args.aug_hflip
    
    if hasattr(args, 'aug_vflip') and args.aug_vflip is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['vertical_flip'] = args.aug_vflip
    
    if hasattr(args, 'aug_rotation') and args.aug_rotation is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['rotation'] = args.aug_rotation
    
    if hasattr(args, 'aug_noise') and args.aug_noise is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['gaussian_noise'] = args.aug_noise
    
    if hasattr(args, 'aug_noise_std') and args.aug_noise_std is not None:
        if 'augmentation' not in config['data']:
            config['data']['augmentation'] = {}
        config['data']['augmentation']['gaussian_noise_std'] = args.aug_noise_std
    
    return config

