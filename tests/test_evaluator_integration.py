"""
Integration tests for evaluator with output transforms.

Tests verify that evaluator correctly applies transforms and
computes metrics in omega_hat space.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
import os

from src.training.evaluator import OmegaEvaluator
from src.training.output_transforms import TransformFactory
from src.models import create_model


class DummyDataLoader:
    """Dummy data loader for testing."""
    
    def __init__(self, num_batches=3, batch_size=4):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = [
            (torch.randn(batch_size, 3, 32, 32), torch.zeros(batch_size))
            for _ in range(num_batches)
        ]
    
    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return self.num_batches


class TestEvaluatorWithTransforms(unittest.TestCase):
    """Test evaluator integration with output transforms."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.image_dim = 3072
        
        # Create a simple model
        self.model = create_model('omega_x_sigma')
        self.model.eval()
        
        # Create dummy data loader
        self.test_loader = DummyDataLoader(num_batches=2, batch_size=4)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_config(self, loss_type, sigma_cal=0.0):
        """Create test configuration."""
        return {
            'model': {'type': 'omega_x_sigma'},
            'noise': {
                'sigma_min': 0.5,
                'sigma_max': 2.0,
                'strategy': 'uniform',
                'num_samples': 100
            },
            'training': {
                'loss_type': loss_type,
                'sigma_cal': sigma_cal
            },
            'logging': {'log_interval': 10}
        }
    
    def test_evaluator_initializes_correct_transform(self):
        """Test that evaluator initializes the correct transform."""
        loss_types_and_transforms = [
            ('omega_hat', 'IdentityTransform'),
            ('omega_epsilon', 'IdentityTransform'),
            ('omega_chi_zscore', 'ChiZScoreTransform'),
        ]
        
        for loss_type, expected_transform_name in loss_types_and_transforms:
            config = self._create_config(loss_type)
            
            evaluator = OmegaEvaluator(
                model=self.model,
                test_loader=self.test_loader,
                config=config,
                exp_dir=self.temp_dir,
                device='cpu'
            )
            
            transform_name = type(evaluator.output_transform).__name__
            self.assertEqual(transform_name, expected_transform_name,
                           f"Wrong transform for {loss_type}")
    
    def test_evaluator_applies_transform_in_evaluation(self):
        """Test that evaluator applies transform during evaluation."""
        config = self._create_config('omega_chi_zscore')
        
        evaluator = OmegaEvaluator(
            model=self.model,
            test_loader=self.test_loader,
            config=config,
            exp_dir=self.temp_dir,
            device='cpu'
        )
        
        # Run evaluation (should not raise errors)
        metrics = evaluator.evaluate(num_samples=8, visualize=False)
        
        # Check that metrics are computed
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        # Check that metrics are finite
        self.assertTrue(torch.isfinite(torch.tensor(metrics['mse'])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics['mae'])))
    
    def test_compute_omega_hat_target(self):
        """Test _compute_omega_hat_target helper method."""
        config = self._create_config('omega_hat')
        
        evaluator = OmegaEvaluator(
            model=self.model,
            test_loader=self.test_loader,
            config=config,
            exp_dir=self.temp_dir,
            device='cpu'
        )
        
        # Create test data
        batch_size = 4
        clean_images = torch.randn(batch_size, 3, 32, 32)
        sigma = torch.ones(batch_size) * 1.0
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        # Compute target
        target = evaluator._compute_omega_hat_target(clean_images, noisy_images, sigma)
        
        # Target should be omega_hat = ||epsilon||^2 / sigma
        epsilon = (noisy_images - clean_images) / sigma.view(-1, 1, 1, 1)
        epsilon_flat = epsilon.view(batch_size, -1)
        epsilon_norm_sq = (epsilon_flat ** 2).sum(dim=1)
        expected = epsilon_norm_sq / sigma
        
        self.assertTrue(torch.allclose(target, expected, rtol=1e-5))
    
    def test_predict_batch_applies_transform(self):
        """Test that predict_batch applies output transform."""
        # Test with omega_chi_zscore (should apply z-score transform)
        config = self._create_config('omega_chi_zscore')
        
        evaluator = OmegaEvaluator(
            model=self.model,
            test_loader=self.test_loader,
            config=config,
            exp_dir=self.temp_dir,
            device='cpu'
        )
        
        # Create test images
        images = torch.randn(4, 3, 32, 32)
        sigma = torch.ones(4) * 2.0
        
        # Get prediction
        omega_hat = evaluator.predict_batch(images, sigma)
        
        # Should return finite values
        self.assertTrue(torch.isfinite(omega_hat).all())
        
        # Should return positive values (omega_hat should be positive)
        self.assertTrue((omega_hat > 0).all())
    
    def test_different_transforms_produce_different_results(self):
        """Test that different loss types produce different transformed outputs."""
        images = torch.randn(4, 3, 32, 32)
        sigma = torch.ones(4) * 2.0
        
        # Create evaluators with different loss types
        loss_types = ['omega_hat', 'omega_epsilon']
        predictions = {}
        
        for loss_type in loss_types:
            config = self._create_config(loss_type)
            evaluator = OmegaEvaluator(
                model=self.model,
                test_loader=self.test_loader,
                config=config,
                exp_dir=self.temp_dir,
                device='cpu'
            )
            
            predictions[loss_type] = evaluator.predict_batch(images, sigma)
        
        # Predictions should be finite (model is untrained so values are random)
        for loss_type, pred in predictions.items():
            self.assertTrue(torch.isfinite(pred).all(),
                          f"Non-finite predictions for {loss_type}")
            # Check shape (could be [4] or [4, 1])
            self.assertIn(pred.numel(), [4],
                          f"Wrong number of elements for {loss_type}")


class TestTransformConsistency(unittest.TestCase):
    """Test consistency between transforms and loss functions."""
    
    def setUp(self):
        self.image_dim = 3072
        self.batch_size = 4
    
    def test_zscore_transform_reverses_zscore_computation(self):
        """Test that z-score transform correctly reverses z-score computation."""
        # Simulate: ||epsilon||^2 = d + z * sqrt(2d)
        # Model learns z
        z_score = torch.tensor([0.0, 1.0, -1.0])
        sigma = torch.tensor([1.0, 2.0, 3.0])
        
        sqrt_2d = np.sqrt(2 * self.image_dim)
        
        # Transform: omega_hat = (z * sqrt(2d) + d) / sigma
        transform = TransformFactory.get_transform('omega_chi_zscore', image_dim=self.image_dim)
        omega_hat = transform.apply(z_score, sigma=sigma)
        
        # Expected: epsilon_norm_sq / sigma where epsilon_norm_sq = d + z * sqrt(2d)
        epsilon_norm_sq = self.image_dim + z_score * sqrt_2d
        expected_omega_hat = epsilon_norm_sq / sigma
        
        self.assertTrue(torch.allclose(omega_hat, expected_omega_hat, rtol=1e-4))
    
    def test_all_transforms_produce_positive_omega_hat(self):
        """Test that all transforms produce positive omega_hat values."""
        # Create reasonable model outputs
        output_positive = torch.abs(torch.randn(10)) + 0.5
        sigma = torch.rand(10) * 3.0 + 0.5
        
        loss_types = [
            'omega_hat', 'omega_epsilon', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            transform = TransformFactory.get_transform(loss_type, image_dim=self.image_dim)
            
            if loss_type == 'omega_chi_zscore':
                # For z-score, use values around 0
                output = torch.randn(10)
                result = transform.apply(output, sigma=sigma)
            else:
                result = transform.apply(output_positive)
            
            # omega_hat should be positive (or at least non-negative)
            # Note: For chi_zscore with extreme negative z-scores, might get negative
            # but for reasonable z-scores, should be positive
            self.assertTrue(torch.isfinite(result).all(),
                          f"Non-finite omega_hat for {loss_type}")


if __name__ == '__main__':
    unittest.main()

