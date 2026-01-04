"""
Main CLI for sigma-score-estimator.

Provides commands for training, testing, and exporting models.
"""

import argparse
import os
import sys
import torch
import random
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import create_model
from src.datasets import get_cifar10_dataloaders
from src.training import OmegaTrainer, OmegaEvaluator
from src.utils import (
    load_config,
    save_config,
    validate_config,
    override_config_with_args,
    create_experiment_dir,
    load_checkpoint,
    load_model_only,
    save_model_only,
    setup_logger,
)


def str_to_bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable cudnn benchmark for better performance with fixed input sizes
        # This is crucial for getting good performance on high-end GPUs like A100
        torch.backends.cudnn.benchmark = True
        # Note: Setting deterministic=True can slow down training significantly
        # Only enable if exact reproducibility is critical
        torch.backends.cudnn.deterministic = False


def train_command(args):
    """Execute training command."""
    print("=" * 80)
    print("TRAINING SIGMA-SCORE-ESTIMATOR")
    print("=" * 80)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Default to config/default.yaml if no config is specified
        default_config_path = 'config/default.yaml'
        if os.path.exists(default_config_path):
            config = load_config(default_config_path)
        else:
            # Fallback to programmatic default if file doesn't exist
            from src.utils.config import create_default_config
            config = create_default_config()
            print(f"Warning: {default_config_path} not found. Using programmatic default config.")
    
    # Override config with command-line arguments
    config = override_config_with_args(config, args)
    
    # Check if trying to train a computational method
    loss_type = config['training'].get('loss_type', 'omega_hat')
    computational_methods = ['omega_edm', 'omega_expected', 'omega_hybrid']
    
    if loss_type in computational_methods:
        print("=" * 80)
        print(f"ERROR: Cannot train computational method '{loss_type}'")
        print("=" * 80)
        
        if loss_type == 'omega_edm':
            print("\nThe 'omega_edm' method is a computational estimator that uses")
            print("a pretrained EDM denoiser. It does not require training.")
            print("\nTo evaluate using the EDM computational method, use:")
            print("  python main.py test --method omega_edm --config config/method_edm.yaml")
        elif loss_type == 'omega_expected':
            print("\nThe 'omega_expected' method is a computational estimator that uses")
            print("the statistical expectation E[||ε||²] = d. It does not require training.")
            print("\nTo evaluate using the expected value computational method, use:")
            print("  python main.py test --method omega_expected --config config/method_expected.yaml")
        elif loss_type == 'omega_hybrid':
            print("\nThe 'omega_hybrid' method is a computational estimator that combines")
            print("EDM denoiser and expected value methods. It does not require training.")
            print("\nTo evaluate using the hybrid computational method, use:")
            print("  python main.py test --method omega_hybrid --config config/method_hybrid.yaml")
        
        print("\nFor trainable methods, use one of:")
        print("  - omega_hat")
        print("  - omega_epsilon")
        print("  - omega_chi_zscore")
        print("=" * 80)
        sys.exit(1)
    
    # Validate configuration
    validate_config(config)
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Create experiment directory under experiments/train by default
    if args.exp_dir:
        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = create_experiment_dir(
            base_dir=os.path.join('experiments', 'train'),
            prefix='exp',
            config=config
        )
    
    print(f"Experiment directory: {exp_dir}")
    
    # Save configuration to experiment directory
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['data']['batch_size'],
        train_val_split=config['data']['train_val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['data']['num_workers'],
        data_root=config['data']['data_root'],
        augmentation_config=config['data'].get('augmentation', None)
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating model: {config['model']['type']}")
    model = create_model(config['model']['type'])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = OmegaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        exp_dir=exp_dir,
        device=device,
        resume_checkpoint=args.resume if hasattr(args, 'resume') else None
    )
    
    # Train
    print("\nStarting training...\n")
    results = trainer.train()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Best validation loss: {results['best_val_loss']:.6f} at epoch {results['best_epoch']}")
    print(f"Final train loss: {results['train_losses'][-1]:.6f}")
    print(f"Final val loss: {results['val_losses'][-1]:.6f}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


def test_command(args):
    """Execute testing command."""
    print("=" * 80)
    print("TESTING SIGMA-SCORE-ESTIMATOR")
    print("=" * 80)
    
    # Determine if using computational method
    computational_methods = ['omega_edm', 'omega_expected', 'omega_hybrid']
    use_computational = hasattr(args, 'method') and args.method in computational_methods
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    elif args.checkpoint:
        # Try to load config from checkpoint directory (flat or legacy layout)
        checkpoint_dir = os.path.dirname(args.checkpoint)
        config_candidates = [
            os.path.join(checkpoint_dir, 'config.yaml'),
            os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml'),
        ]
        config = None
        for candidate in config_candidates:
            if os.path.exists(candidate):
                config = load_config(candidate)
                break
        if config is None:
            print("Warning: No config found. Using default config.")
            from src.utils.config import create_default_config
            config = create_default_config()
    elif use_computational:
        # For computational methods, config is required
        method_name = args.method if hasattr(args, 'method') else 'computational'
        print(f"Error: --config is required when using computational methods (--method {method_name})")
        sys.exit(1)
    else:
        raise ValueError("Must provide either --config or --checkpoint")
    
    # Override config with command-line arguments
    config = override_config_with_args(config, args)
    
    # If method is specified, override loss_type in config
    if use_computational:
        method_name = args.method
        config['training']['loss_type'] = method_name
        config['model']['type'] = method_name
        print(f"Using computational method: {method_name}")
    
    # Set device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory for test results
    if args.exp_dir:
        exp_dir = args.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = create_experiment_dir(
            base_dir=os.path.join('experiments', 'test'),
            prefix='exp',
            config=config
        )
    
    print(f"Test results directory: {exp_dir}")
    
    # Save configuration
    config_save_path = os.path.join(exp_dir, 'config.yaml')
    save_config(config, config_save_path)
    
    # Load data (no augmentation for testing)
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        batch_size=config['data']['batch_size'],
        train_val_split=config['data']['train_val_split'],
        test_split=config['data']['test_split'],
        num_workers=config['data']['num_workers'],
        data_root=config['data']['data_root'],
        augmentation_config=None  # No augmentation for testing
    )
    print(f"Test batches: {len(test_loader)}")
    
    # Load or create model
    if use_computational:
        # For computational methods, model will be created by evaluator
        method_name = args.method if hasattr(args, 'method') else 'computational'
        print(f"\nUsing {method_name} computational estimator (no checkpoint required)")
        model = None
    else:
        # Load trained model from checkpoint
        print(f"\nLoading model from: {args.checkpoint}")
        
        # Create model architecture
        model = create_model(config['model']['type'])
        
        # Load checkpoint
        model = load_model_only(args.checkpoint, model=model, device=device)
        
        print(f"Model loaded successfully")
        print(f"Model type: {config['model']['type']}")
    
    # Create evaluator
    print("\nInitializing evaluator...")
    evaluator = OmegaEvaluator(
        model=model,
        test_loader=test_loader,
        config=config,
        exp_dir=exp_dir,
        device=device
    )
    
    # Evaluate
    print("\nStarting evaluation...\n")
    metrics = evaluator.evaluate(
        num_samples=args.test_samples if hasattr(args, 'test_samples') else None,
        visualize=True
    )
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED")
    print("=" * 80)
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2_score']:.6f}")
    print(f"Mean Relative Error: {metrics['mean_relative_error']:.4f}")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


def export_command(args):
    """Execute export command."""
    print("=" * 80)
    print("EXPORTING MODEL")
    print("=" * 80)
    
    if not args.checkpoint:
        raise ValueError("Must provide --checkpoint for export")
    
    if not args.output:
        raise ValueError("Must provide --output for export")
    
    # Set device
    device = 'cpu'  # Export to CPU for portability
    
    # Try to infer model type from checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    config_candidates = [
        os.path.join(checkpoint_dir, 'config.yaml'),
        os.path.join(os.path.dirname(checkpoint_dir), 'config.yaml'),
    ]
    
    config = None
    for candidate in config_candidates:
        if os.path.exists(candidate):
            config = load_config(candidate)
            break
    
    if config is not None:
        model_type = config['model']['type']
    elif args.model_type:
        model_type = args.model_type
    else:
        raise ValueError("Cannot infer model type. Please provide --model-type")
    
    print(f"Model type: {model_type}")
    print(f"Loading from: {args.checkpoint}")
    
    # Create model
    model = create_model(model_type)
    
    # Load checkpoint
    model = load_model_only(args.checkpoint, model=model, device=device)
    
    # Save model
    output_format = args.output.split('.')[-1]
    if output_format not in ['pkl', 'pth', 'pt']:
        print(f"Warning: Unknown format '{output_format}'. Using 'pkl'")
        output_format = 'pkl'
    
    save_model_only(model, args.output, save_format=output_format)
    
    print("\n" + "=" * 80)
    print("EXPORT COMPLETED")
    print("=" * 80)
    print(f"Model exported to: {args.output}")
    print(f"Format: {output_format}")
    print("=" * 80)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Sigma-Score-Estimator: Train models to estimate noise-level score gradient',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--model-type', type=str, choices=['omega_x', 'omega_x_sigma'],
                             help='Model type')
    train_parser.add_argument('--loss-type', type=str, 
                             choices=['omega_hat', 'omega_epsilon', 'omega_chi_zscore'],
                             help='Loss function type (note: omega_edm cannot be trained)')
    train_parser.add_argument('--noise-strategy', type=str,
                             choices=['uniform', 'log_uniform', 'select_batch'],
                             help='Noise sampling strategy')
    train_parser.add_argument('--sigma-min', type=float, help='Minimum noise level')
    train_parser.add_argument('--sigma-max', type=float, help='Maximum noise level')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'],
                             help='Optimizer type')
    train_parser.add_argument('--weight-decay', type=float, help='Weight decay')
    train_parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'none'],
                             help='Learning rate scheduler')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    train_parser.add_argument('--seed', type=int, help='Random seed')
    train_parser.add_argument('--exp-dir', type=str, help='Experiment directory')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--early-stopping', type=str_to_bool, help='Enable/disable early stopping (true/false)')
    
    # Data augmentation arguments
    train_parser.add_argument('--augmentation', type=str_to_bool, help='Enable/disable data augmentation (true/false)')
    train_parser.add_argument('--aug-hflip', type=str_to_bool, help='Enable/disable horizontal flip augmentation')
    train_parser.add_argument('--aug-vflip', type=str_to_bool, help='Enable/disable vertical flip augmentation')
    train_parser.add_argument('--aug-rotation', type=str_to_bool, help='Enable/disable rotation augmentation')
    train_parser.add_argument('--aug-noise', type=str_to_bool, help='Enable/disable Gaussian noise augmentation')
    train_parser.add_argument('--aug-noise-std', type=float, help='Standard deviation for Gaussian noise augmentation')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a trained model or computational method')
    test_parser.add_argument('--checkpoint', type=str,
                            help='Path to model checkpoint (not required for computational methods)')
    test_parser.add_argument('--config', type=str, help='Path to config file (required for computational methods)')
    test_parser.add_argument('--method', type=str, choices=['omega_edm', 'omega_expected', 'omega_hybrid'],
                            help='Use computational method instead of trained model')
    test_parser.add_argument('--test-samples', type=int, help='Number of samples to test')
    test_parser.add_argument('--sigma-min', type=float, help='Minimum noise level')
    test_parser.add_argument('--sigma-max', type=float, help='Maximum noise level')
    test_parser.add_argument('--sigma-threshold', type=float,
                            help='Sigma threshold for hybrid method (only for omega_hybrid)')
    test_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    test_parser.add_argument('--exp-dir', type=str, help='Test results directory')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a trained model')
    export_parser.add_argument('--checkpoint', type=str, required=True,
                               help='Path to model checkpoint')
    export_parser.add_argument('--output', type=str, required=True,
                              help='Output path for exported model')
    export_parser.add_argument('--model-type', type=str, choices=['omega_x', 'omega_x_sigma'],
                               help='Model type (if cannot infer from checkpoint)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'test':
        test_command(args)
    elif args.command == 'export':
        export_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

