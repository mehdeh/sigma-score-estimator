# Unit Tests for Sigma Score Estimator

This directory contains comprehensive unit tests for the output transform refactoring and new loss type implementation.

## Test Files

### 1. `test_output_transforms.py`
Tests for output transformation module:
- **TestIdentityTransform**: Verifies no-op transformation for loss types 1-4
- **TestSigmaToOmegaTransform**: Tests d/sigma transformation for loss types 5-7
- **TestSigmaCalibratedTransform**: Tests calibrated transformation for loss type 8
- **TestChiZScoreTransform**: Tests z-score transformation for loss type 9 (NEW)
- **TestTransformFactory**: Tests factory pattern for creating transforms
- **TestTransformIntegration**: Integration tests with realistic data

### 2. `test_loss_functions.py`
Tests for refactored and new loss functions:
- **TestSigmaDirectLoss**: Verifies refactored loss type 5 (no internal transform)
- **TestSigmaNormalizedLoss**: Verifies refactored loss type 6
- **TestSigmaRelativeLoss**: Verifies refactored loss type 7
- **TestSigmaCalibratedLoss**: Verifies refactored loss type 8
- **TestOmegaChiZScoreLoss**: Tests NEW loss type 9 (chi-squared z-score)
- **TestLossFactory**: Tests loss factory for all 9 loss types
- **TestLossIntegration**: Integration tests (finite values, backprop, etc.)

### 3. `test_evaluator_integration.py`
Integration tests for evaluator with transforms:
- **TestEvaluatorWithTransforms**: Tests evaluator applies correct transforms
- **TestTransformConsistency**: Verifies transforms are inverse of what losses learn

## Running Tests

### Run All Tests

```bash
# From project root
cd /home/ubuntu/repos/sigma-score-estimator

# Activate conda environment
eval "$(conda shell.bash hook)" && conda activate myenv

# Run all tests
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
# Test output transforms
python -m pytest tests/test_output_transforms.py -v

# Test loss functions
python -m pytest tests/test_loss_functions.py -v

# Test evaluator integration
python -m pytest tests/test_evaluator_integration.py -v
```

### Run Specific Test Class

```bash
# Test only transform factory
python -m pytest tests/test_output_transforms.py::TestTransformFactory -v

# Test only new loss type 9
python -m pytest tests/test_loss_functions.py::TestOmegaChiZScoreLoss -v
```

### Run with Coverage

```bash
# Install coverage if needed
pip install pytest-cov

# Run with coverage report
python -m pytest tests/ --cov=src/training --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

### Alternative: Run with unittest

```bash
# Run single file
python -m unittest tests.test_output_transforms

# Run all tests
python -m unittest discover tests/

# Run with verbose output
python -m unittest discover tests/ -v
```

## Test Coverage

The tests cover:

### ✅ Output Transforms
- Identity transform (no change)
- Sigma to omega transform (d / sigma)
- Calibrated sigma transform (d / (sigma_cal - output))
- Chi-squared z-score transform ((z * sqrt(2d) + d) / sigma)
- Transform factory for all 9 loss types
- Edge cases (division by zero, negative values, etc.)

### ✅ Loss Functions
- Refactored loss types 5-8 (no internal transforms)
- New loss type 9 (chi-squared z-score)
- Perfect prediction (zero loss)
- Finite and differentiable outputs
- Correct mathematical formulations

### ✅ Integration
- Evaluator initializes correct transform
- Transform applied during evaluation
- Consistent omega_hat space evaluation
- predict_batch applies transforms
- Transform-loss consistency

## Expected Results

All tests should pass. If any test fails:

1. **Check the error message** - it will indicate which test failed and why
2. **Verify your implementation** - compare with the mathematical formulas in MATHEMATICAL_BACKGROUND.md
3. **Check for NaN/Inf values** - ensure all computations are numerically stable
4. **Review transforms** - ensure transforms are applied at the right stage (inference, not training)

## Test Statistics

- **Total test files**: 3
- **Total test classes**: 15
- **Total test methods**: ~50+
- **Code coverage**: Targets >90% for modified files

## Adding New Tests

When adding new loss types or transforms:

1. Add tests to appropriate file
2. Follow existing test patterns
3. Test both perfect and imperfect predictions
4. Test edge cases (zero, negative, extreme values)
5. Test integration with evaluator

Example test structure:

```python
class TestNewLossType(unittest.TestCase):
    def setUp(self):
        # Initialize loss function
        pass
    
    def test_perfect_prediction(self):
        # Test zero loss for perfect prediction
        pass
    
    def test_finite_output(self):
        # Test outputs are finite
        pass
    
    def test_differentiable(self):
        # Test gradients exist and are finite
        pass
```

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root
cd /home/ubuntu/repos/sigma-score-estimator

# Run tests as module
python -m pytest tests/
```

### CUDA Errors

Tests run on CPU by default. If you encounter CUDA errors:
- Check that `device='cpu'` is set in test configs
- Ensure tensors are moved to CPU before assertions

### Numerical Precision

Some tests use `assertAlmostEqual` with tolerance:
- Default: 6 decimal places
- For transforms: 4-5 places (due to floating point)
- Adjust `places=` parameter if needed

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    conda activate myenv
    python -m pytest tests/ -v --cov=src/training
```

## Questions?

See:
- [MATHEMATICAL_BACKGROUND.md](../MATHEMATICAL_BACKGROUND.md) for mathematical details
- [USAGE_GUIDE.md](../USAGE_GUIDE.md) for usage examples
- Implementation plan for architectural details

