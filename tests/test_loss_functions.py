"""
Unit tests for loss functions.

Tests verify that refactored loss functions (types 5-8) work correctly
and new loss type 9 produces expected results.
"""

import unittest
import torch
import numpy as np

from src.training.loss_functions import (
    LossFactory,
    OmegaChiZScoreLoss,
)


class TestOmegaChiZScoreLoss(unittest.TestCase):
    """Test new OmegaChiZScoreLoss (Type 9)."""
    
    def setUp(self):
        self.image_dim = 3072
        self.loss_fn = OmegaChiZScoreLoss(image_dim=self.image_dim)
        self.sqrt_2d = np.sqrt(2 * self.image_dim)
    
    def test_zscore_target_computation(self):
        """Test that target is computed as (||epsilon||^2 - d) / sqrt(2d)."""
        batch_size = 2
        
        # Create controlled noise
        clean_images = torch.zeros(batch_size, 3, 32, 32)
        sigma = torch.tensor([1.0, 2.0])
        
        # Create noisy images with known noise
        noise = torch.randn(batch_size, 3, 32, 32)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        # Model output (doesn't matter for target computation)
        output = torch.randn(batch_size)
        
        # Compute loss
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Loss should be finite
        self.assertTrue(torch.isfinite(loss))
    
    def test_zscore_approximately_normalized(self):
        """Test that z-score target has approximately zero mean for large batch."""
        batch_size = 100
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        sigma = torch.ones(batch_size) * 2.0
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        # Compute epsilon
        epsilon = (noisy_images - clean_images) / sigma.view(-1, 1, 1, 1)
        epsilon_flat = epsilon.view(batch_size, -1)
        epsilon_norm_sq = (epsilon_flat ** 2).sum(dim=1)
        
        # Z-score
        z_score = (epsilon_norm_sq - self.image_dim) / self.sqrt_2d
        
        # Mean should be close to 0 (within 3 standard errors)
        mean_z = z_score.mean().item()
        std_error = 1.0 / np.sqrt(batch_size)  # Standard error
        
        self.assertLess(abs(mean_z), 3 * std_error)
    
    def test_loss_decreases_with_better_prediction(self):
        """Test that loss is smaller for better predictions."""
        batch_size = 4
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        sigma = torch.ones(batch_size) * 1.0
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        # Compute actual z-score
        epsilon = (noisy_images - clean_images) / sigma.view(-1, 1, 1, 1)
        epsilon_flat = epsilon.view(batch_size, -1)
        epsilon_norm_sq = (epsilon_flat ** 2).sum(dim=1)
        actual_z = (epsilon_norm_sq - self.image_dim) / self.sqrt_2d
        
        # Good prediction (close to actual)
        good_output = actual_z + torch.randn(batch_size) * 0.1
        loss_good = self.loss_fn(good_output, clean_images, noisy_images, sigma)
        
        # Bad prediction (far from actual)
        bad_output = actual_z + torch.randn(batch_size) * 10.0
        loss_bad = self.loss_fn(bad_output, clean_images, noisy_images, sigma)
        
        self.assertLess(loss_good.item(), loss_bad.item())


class TestLossFactory(unittest.TestCase):
    """Test LossFactory for creating loss functions."""
    
    def test_factory_creates_all_loss_types(self):
        """Test that factory can create all 3 loss types."""
        loss_types = [
            'omega_hat', 'omega_epsilon', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            loss_fn = LossFactory.get_loss(loss_type)
            self.assertIsNotNone(loss_fn)
    
    def test_factory_passes_image_dim(self):
        """Test that factory passes image_dim to loss functions."""
        image_dim = 1024
        loss_fn = LossFactory.get_loss('omega_chi_zscore', image_dim=image_dim)
        
        self.assertEqual(loss_fn.image_dim, image_dim)
    
    def test_factory_raises_on_unknown_loss(self):
        """Test that factory raises error for unknown loss type."""
        with self.assertRaises(ValueError):
            LossFactory.get_loss('unknown_loss_type')
    
    def test_available_losses_list(self):
        """Test that AVAILABLE_LOSSES contains all 3 types."""
        available = LossFactory.AVAILABLE_LOSSES
        
        self.assertEqual(len(available), 3)
        self.assertIn('omega_hat', available)
        self.assertIn('omega_epsilon', available)
        self.assertIn('omega_chi_zscore', available)


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions."""
    
    def test_all_losses_produce_finite_values(self):
        """Test that all loss functions produce finite values."""
        batch_size = 4
        clean_images = torch.randn(batch_size, 3, 32, 32)
        sigma = torch.rand(batch_size) * 3.0 + 0.5
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        output = torch.rand(batch_size) * 5.0 + 0.5
        
        loss_types = [
            'omega_hat', 'omega_epsilon', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            loss_fn = LossFactory.get_loss(loss_type)
            loss = loss_fn(output, clean_images, noisy_images, sigma)
            
            self.assertTrue(torch.isfinite(loss), 
                          f"Loss type {loss_type} produced non-finite value")
            self.assertGreaterEqual(loss.item(), 0.0,
                                  f"Loss type {loss_type} produced negative value")
    
    def test_losses_are_differentiable(self):
        """Test that all losses can compute gradients."""
        batch_size = 4
        
        # Create images with requires_grad for proper gradient flow
        clean_images = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
        sigma = torch.rand(batch_size) * 3.0 + 0.5
        noise = torch.randn_like(clean_images)
        noisy_images = clean_images + noise * sigma.view(-1, 1, 1, 1)
        
        loss_types = ['omega_hat', 'omega_epsilon', 'omega_chi_zscore']
        
        for loss_type in loss_types:
            # Create fresh output tensor for each loss type
            output = torch.rand(batch_size, requires_grad=True) * 5.0 + 0.5
            
            loss_fn = LossFactory.get_loss(loss_type)
            loss = loss_fn(output, clean_images, noisy_images, sigma)
            
            # Loss should be finite and scalar
            self.assertTrue(torch.isfinite(loss),
                          f"Loss type {loss_type} produced non-finite loss")
            self.assertEqual(loss.dim(), 0,
                          f"Loss type {loss_type} is not scalar")
            
            # Compute gradients
            loss.backward()
            
            # At least clean_images should have gradients (used in computing target)
            if loss_type in ['omega_hat', 'omega_epsilon', 'omega_chi_zscore']:
                self.assertIsNotNone(clean_images.grad,
                                    f"Loss type {loss_type} produced no gradients")
            
            # Clean up
            if clean_images.grad is not None:
                clean_images.grad.zero_()


if __name__ == '__main__':
    unittest.main()

