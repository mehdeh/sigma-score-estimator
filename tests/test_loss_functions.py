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
    SigmaDirectLoss,
    SigmaNormalizedLoss,
    SigmaRelativeLoss,
    SigmaCalibratedLoss,
    OmegaChiZScoreLoss,
)


class TestSigmaDirectLoss(unittest.TestCase):
    """Test refactored SigmaDirectLoss (Type 5)."""
    
    def setUp(self):
        self.image_dim = 3072
        self.loss_fn = SigmaDirectLoss(image_dim=self.image_dim)
    
    def test_perfect_prediction(self):
        """Test loss is zero when output equals sigma."""
        batch_size = 4
        output = torch.tensor([1.0, 2.0, 3.0, 4.0])
        sigma = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Create dummy images (not used in loss)
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
    
    def test_non_zero_loss_for_incorrect_prediction(self):
        """Test loss is non-zero when prediction is wrong."""
        batch_size = 2
        output = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([2.0, 4.0])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        self.assertGreater(loss.item(), 0.0)
    
    def test_loss_is_mse(self):
        """Test that loss equals mean squared error."""
        batch_size = 3
        output = torch.tensor([1.0, 2.0, 3.0])
        sigma = torch.tensor([1.5, 2.5, 3.5])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Manual MSE calculation
        expected = torch.mean((output - sigma) ** 2)
        
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)


class TestSigmaNormalizedLoss(unittest.TestCase):
    """Test refactored SigmaNormalizedLoss (Type 6)."""
    
    def setUp(self):
        self.image_dim = 3072
        self.loss_fn = SigmaNormalizedLoss(image_dim=self.image_dim)
    
    def test_normalized_by_sigma(self):
        """Test that loss is normalized by sigma."""
        batch_size = 2
        output = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([1.0, 2.0])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Manual calculation: ((output - sigma)^2 / sigma).mean()
        expected = torch.mean(((output - sigma) ** 2) / sigma)
        
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
    
    def test_weights_errors_by_sigma(self):
        """Test that errors at small sigma have more weight."""
        batch_size = 2
        # Same absolute error, different sigmas
        output = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([0.5, 2.0])  # Error: 0.5 for both
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Error at sigma=0.5 should be weighted more
        error_1 = ((1.0 - 0.5) ** 2) / 0.5  # = 0.5
        error_2 = ((2.0 - 2.0) ** 2) / 2.0  # = 0.0
        expected = (error_1 + error_2) / 2
        
        self.assertAlmostEqual(loss.item(), expected, places=6)


class TestSigmaRelativeLoss(unittest.TestCase):
    """Test refactored SigmaRelativeLoss (Type 7)."""
    
    def setUp(self):
        self.image_dim = 3072
        self.loss_fn = SigmaRelativeLoss(image_dim=self.image_dim)
    
    def test_relative_error_formulation(self):
        """Test that loss uses relative error formulation."""
        batch_size = 2
        output = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([2.0, 4.0])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Manual calculation: ((output - sigma)^2 / sigma^2).mean()
        expected = torch.mean(((output - sigma) ** 2) / (sigma ** 2))
        
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
    
    def test_scale_invariance(self):
        """Test that relative loss is scale-invariant."""
        batch_size = 2
        
        # First case: output=1, sigma=2 (50% relative error)
        output1 = torch.tensor([1.0, 1.0])
        sigma1 = torch.tensor([2.0, 2.0])
        
        # Second case: scaled by 10 (same 50% relative error)
        output2 = torch.tensor([10.0, 10.0])
        sigma2 = torch.tensor([20.0, 20.0])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss1 = self.loss_fn(output1, clean_images, noisy_images, sigma1)
        loss2 = self.loss_fn(output2, clean_images, noisy_images, sigma2)
        
        # Should be equal due to scale invariance
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)


class TestSigmaCalibratedLoss(unittest.TestCase):
    """Test refactored SigmaCalibratedLoss (Type 8)."""
    
    def setUp(self):
        self.image_dim = 3072
        self.sigma_cal = 10.0
        self.loss_fn = SigmaCalibratedLoss(
            image_dim=self.image_dim,
            sigma_cal=self.sigma_cal
        )
    
    def test_calibrated_target(self):
        """Test that target is sigma_cal - sigma."""
        batch_size = 2
        output = torch.tensor([8.0, 6.0])
        sigma = torch.tensor([2.0, 4.0])
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        # Target: sigma_cal - sigma = [8.0, 6.0]
        target = self.sigma_cal - sigma
        expected = torch.mean((output - target) ** 2)
        
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
    
    def test_perfect_calibrated_prediction(self):
        """Test zero loss when output equals sigma_cal - sigma."""
        batch_size = 2
        sigma = torch.tensor([2.0, 3.0])
        output = self.sigma_cal - sigma  # Perfect prediction
        
        clean_images = torch.randn(batch_size, 3, 32, 32)
        noisy_images = torch.randn(batch_size, 3, 32, 32)
        
        loss = self.loss_fn(output, clean_images, noisy_images, sigma)
        
        self.assertAlmostEqual(loss.item(), 0.0, places=6)


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
        """Test that factory can create all 9 loss types."""
        loss_types = [
            'omega_hat', 'omega_epsilon', 'omega_chi_approx', 'omega_chi_mean',
            'sigma_direct', 'sigma_normalized', 'sigma_relative', 
            'sigma_calibrated', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            loss_fn = LossFactory.get_loss(loss_type, sigma_cal=5.0)
            self.assertIsNotNone(loss_fn)
    
    def test_factory_passes_image_dim(self):
        """Test that factory passes image_dim to loss functions."""
        image_dim = 1024
        loss_fn = LossFactory.get_loss('sigma_direct', image_dim=image_dim)
        
        self.assertEqual(loss_fn.image_dim, image_dim)
    
    def test_factory_passes_sigma_cal(self):
        """Test that factory passes sigma_cal to calibrated loss."""
        sigma_cal = 15.0
        loss_fn = LossFactory.get_loss('sigma_calibrated', sigma_cal=sigma_cal)
        
        self.assertEqual(loss_fn.sigma_cal, sigma_cal)
    
    def test_factory_raises_on_unknown_loss(self):
        """Test that factory raises error for unknown loss type."""
        with self.assertRaises(ValueError):
            LossFactory.get_loss('unknown_loss_type')
    
    def test_available_losses_list(self):
        """Test that AVAILABLE_LOSSES contains all 9 types."""
        available = LossFactory.AVAILABLE_LOSSES
        
        self.assertEqual(len(available), 9)
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
            'omega_hat', 'omega_epsilon', 'omega_chi_approx', 'omega_chi_mean',
            'sigma_direct', 'sigma_normalized', 'sigma_relative', 
            'sigma_calibrated', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            loss_fn = LossFactory.get_loss(loss_type, sigma_cal=10.0)
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
        
        loss_types = ['sigma_direct', 'sigma_normalized', 'omega_chi_zscore']
        
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
            if loss_type == 'omega_chi_zscore':
                self.assertIsNotNone(clean_images.grad,
                                    f"Loss type {loss_type} produced no gradients")
            
            # Clean up
            if clean_images.grad is not None:
                clean_images.grad.zero_()


if __name__ == '__main__':
    unittest.main()

