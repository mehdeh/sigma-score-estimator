"""
Unit tests for output transformation module.

Tests verify that all output transforms work correctly and produce
expected results for converting model outputs to omega_hat.
"""

import unittest
import torch
import numpy as np

from src.training.output_transforms import (
    IdentityTransform,
    SigmaToOmegaTransform,
    SigmaCalibratedTransform,
    ChiZScoreTransform,
    TransformFactory,
)


class TestIdentityTransform(unittest.TestCase):
    """Test identity transformation (no change)."""
    
    def setUp(self):
        self.transform = IdentityTransform()
    
    def test_identity_returns_same_output(self):
        """Test that identity transform returns input unchanged."""
        output = torch.tensor([1.0, 2.0, 3.0])
        result = self.transform.apply(output)
        
        self.assertTrue(torch.allclose(result, output))
    
    def test_identity_with_sigma(self):
        """Test that identity ignores sigma parameter."""
        output = torch.tensor([5.0, 10.0])
        sigma = torch.tensor([1.0, 2.0])
        result = self.transform.apply(output, sigma=sigma)
        
        self.assertTrue(torch.allclose(result, output))


class TestSigmaToOmegaTransform(unittest.TestCase):
    """Test sigma to omega_hat transformation."""
    
    def setUp(self):
        self.image_dim = 3072
        self.transform = SigmaToOmegaTransform(image_dim=self.image_dim)
    
    def test_basic_transformation(self):
        """Test d / sigma transformation."""
        # If model predicts sigma = 2.0
        output = torch.tensor([2.0])
        expected = self.image_dim / 2.0  # omega_hat = d / sigma
        
        result = self.transform.apply(output)
        
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_batch_transformation(self):
        """Test transformation on batch."""
        output = torch.tensor([1.0, 2.0, 4.0])
        expected = torch.tensor([
            self.image_dim / 1.0,
            self.image_dim / 2.0,
            self.image_dim / 4.0
        ])
        
        result = self.transform.apply(output)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))
    
    def test_handles_negative_output(self):
        """Test that transformation uses absolute value."""
        output = torch.tensor([-2.0, 2.0])
        
        result = self.transform.apply(output)
        
        # Both should give same result (abs value used)
        self.assertAlmostEqual(result[0].item(), result[1].item(), places=5)
    
    def test_epsilon_prevents_division_by_zero(self):
        """Test that epsilon prevents division by zero."""
        output = torch.tensor([0.0])
        
        # Should not raise error
        result = self.transform.apply(output)
        
        # Result should be finite
        self.assertTrue(torch.isfinite(result).all())


class TestSigmaCalibratedTransform(unittest.TestCase):
    """Test calibrated sigma transformation."""
    
    def setUp(self):
        self.image_dim = 3072
        self.sigma_cal = 10.0
        self.transform = SigmaCalibratedTransform(
            image_dim=self.image_dim,
            sigma_cal=self.sigma_cal
        )
    
    def test_calibrated_transformation(self):
        """Test d / (sigma_cal - output) transformation."""
        # Model predicts (sigma_cal - sigma) = 8.0
        # So sigma = 10 - 8 = 2.0
        # omega_hat = d / sigma = d / 2.0
        output = torch.tensor([8.0])
        expected = self.image_dim / 2.0
        
        result = self.transform.apply(output)
        
        self.assertAlmostEqual(result.item(), expected, places=5)
    
    def test_batch_calibrated_transformation(self):
        """Test calibrated transformation on batch."""
        # Model predicts sigma_cal - sigma
        output = torch.tensor([9.0, 8.0, 5.0])
        # Actual sigmas: 1.0, 2.0, 5.0
        expected = torch.tensor([
            self.image_dim / 1.0,
            self.image_dim / 2.0,
            self.image_dim / 5.0
        ])
        
        result = self.transform.apply(output)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5))


class TestChiZScoreTransform(unittest.TestCase):
    """Test chi-squared z-score transformation."""
    
    def setUp(self):
        self.image_dim = 3072
        self.transform = ChiZScoreTransform(image_dim=self.image_dim)
        self.sqrt_2d = np.sqrt(2 * self.image_dim)
    
    def test_basic_zscore_transformation(self):
        """Test (z * sqrt(2d) + d) / sigma transformation."""
        # Model predicts z-score = 1.0
        output = torch.tensor([1.0])
        sigma = torch.tensor([2.0])
        
        # omega_hat = (1.0 * sqrt(2d) + d) / 2.0
        expected = (1.0 * self.sqrt_2d + self.image_dim) / 2.0
        
        result = self.transform.apply(output, sigma=sigma)
        
        self.assertAlmostEqual(result.item(), float(expected), places=3)
    
    def test_zscore_batch_transformation(self):
        """Test z-score transformation on batch."""
        output = torch.tensor([0.0, 1.0, -1.0])
        sigma = torch.tensor([1.0, 2.0, 3.0])
        
        expected = torch.tensor([
            (0.0 * self.sqrt_2d + self.image_dim) / 1.0,
            (1.0 * self.sqrt_2d + self.image_dim) / 2.0,
            (-1.0 * self.sqrt_2d + self.image_dim) / 3.0
        ], dtype=torch.float32)
        
        result = self.transform.apply(output, sigma=sigma)
        
        self.assertTrue(torch.allclose(result, expected, rtol=1e-3))
    
    def test_requires_sigma(self):
        """Test that transformation requires sigma parameter."""
        output = torch.tensor([1.0])
        
        with self.assertRaises(ValueError):
            self.transform.apply(output, sigma=None)
    
    def test_zero_zscore_gives_expected_value(self):
        """Test that z-score of 0 gives omega_hat = d / sigma."""
        output = torch.tensor([0.0])
        sigma = torch.tensor([2.0])
        
        # When z = 0: omega_hat = d / sigma
        expected = self.image_dim / 2.0
        
        result = self.transform.apply(output, sigma=sigma)
        
        self.assertAlmostEqual(result.item(), expected, places=4)


class TestTransformFactory(unittest.TestCase):
    """Test transform factory for creating appropriate transforms."""
    
    def test_factory_creates_identity_for_omega_hat(self):
        """Test factory returns IdentityTransform for loss types 1-4."""
        for loss_type in ['omega_hat', 'omega_epsilon', 'omega_chi_approx', 'omega_chi_mean']:
            transform = TransformFactory.get_transform(loss_type)
            self.assertIsInstance(transform, IdentityTransform)
    
    def test_factory_creates_sigma_transform_for_types_5_7(self):
        """Test factory returns SigmaToOmegaTransform for loss types 5-7."""
        for loss_type in ['sigma_direct', 'sigma_normalized', 'sigma_relative']:
            transform = TransformFactory.get_transform(loss_type)
            self.assertIsInstance(transform, SigmaToOmegaTransform)
    
    def test_factory_creates_calibrated_for_type_8(self):
        """Test factory returns SigmaCalibratedTransform for loss type 8."""
        transform = TransformFactory.get_transform('sigma_calibrated', sigma_cal=5.0)
        self.assertIsInstance(transform, SigmaCalibratedTransform)
    
    def test_factory_creates_zscore_for_type_9(self):
        """Test factory returns ChiZScoreTransform for loss type 9."""
        transform = TransformFactory.get_transform('omega_chi_zscore')
        self.assertIsInstance(transform, ChiZScoreTransform)
    
    def test_factory_with_custom_image_dim(self):
        """Test factory respects custom image dimension."""
        image_dim = 1024
        transform = TransformFactory.get_transform('sigma_direct', image_dim=image_dim)
        
        self.assertEqual(transform.image_dim, image_dim)
    
    def test_factory_raises_on_unknown_loss_type(self):
        """Test factory raises error for unknown loss type."""
        with self.assertRaises(ValueError):
            TransformFactory.get_transform('unknown_loss_type')
    
    def test_get_available_transforms(self):
        """Test getting list of available transforms."""
        available = TransformFactory.get_available_transforms()
        
        self.assertIn('omega_hat', available)
        self.assertIn('sigma_direct', available)
        self.assertIn('omega_chi_zscore', available)
        self.assertEqual(len(available), 9)


class TestTransformIntegration(unittest.TestCase):
    """Integration tests for transforms with realistic data."""
    
    def test_all_transforms_produce_finite_output(self):
        """Test that all transforms produce finite outputs."""
        batch_size = 10
        output = torch.randn(batch_size).abs() + 0.1  # Positive values
        sigma = torch.rand(batch_size) * 5.0 + 0.1  # Sigma in [0.1, 5.1]
        
        loss_types = [
            'omega_hat', 'omega_epsilon', 'omega_chi_approx', 'omega_chi_mean',
            'sigma_direct', 'sigma_normalized', 'sigma_relative', 
            'sigma_calibrated', 'omega_chi_zscore'
        ]
        
        for loss_type in loss_types:
            transform = TransformFactory.get_transform(loss_type, sigma_cal=10.0)
            
            if loss_type == 'omega_chi_zscore':
                result = transform.apply(output, sigma=sigma)
            else:
                result = transform.apply(output)
            
            self.assertTrue(torch.isfinite(result).all(), 
                          f"Transform {loss_type} produced non-finite values")
    
    def test_transform_output_shapes(self):
        """Test that transforms preserve output shapes."""
        for shape in [(10,), (10, 1)]:
            output = torch.randn(shape).abs() + 0.1
            sigma = torch.rand(10) + 0.1
            
            transform = TransformFactory.get_transform('sigma_direct')
            result = transform.apply(output)
            
            # Should squeeze to 1D
            self.assertEqual(result.shape, (10,))


if __name__ == '__main__':
    unittest.main()

