# Usage Guide

This comprehensive guide covers everything you need to know to train, evaluate, and use the Sigma-Score-Estimator models.

## Table of Contents

1. [Installation](#installation)
2. [Training Models](#training-models)
3. [Testing Models](#testing-models)
4. [Using Trained Models](#using-trained-models)
5. [Configuration Guide](#configuration-guide)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Setup

```bash
# Navigate to the repository
cd /home/ubuntu/repos/sigma-score-estimator

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Training Models

### Basic Training

Train with default configuration:

```bash
python main.py train --config config/default.yaml
```

### Training omega(x, sigma) Model

Train the model that takes both image and noise level as input:

```bash
python main.py train \
    --config config/model_x_sigma.yaml \
    --epochs 100 \
    --batch-size 128 \
    --learning-rate 0.001
```

### Training omega(x) Model

Train the model that takes only the image as input:

```bash
python main.py train \
    --config config/model_x.yaml \
    --epochs 100
```

### Custom Training Configuration

Override specific parameters:

```bash
python main.py train \
    --config config/default.yaml \
    --model-type omega_x_sigma \
    --loss-type omega_hat \
    --noise-strategy log_uniform \
    --sigma-min 0.01 \
    --sigma-max 10.0 \
    --epochs 150 \
    --learning-rate 0.0005 \
    --early-stopping true \
    --device cuda \
    --seed 42
```

### Early Stopping Control

Enable or disable early stopping via CLI:

```bash
# Enable early stopping (default behavior)
python main.py train --config config/default.yaml --early-stopping true

# Disable early stopping
python main.py train --config config/default.yaml --early-stopping false
```

When enabled, training will stop early if validation loss doesn't improve for `patience` epochs (configured in the config file).

### Optimizer and Scheduler Configuration

Configure optimizer and learning rate scheduler via CLI:

```bash
# Use SGD optimizer instead of Adam
python main.py train \
    --config config/default.yaml \
    --optimizer sgd \
    --learning-rate 0.01 \
    --weight-decay 0.0001

# Use step scheduler instead of cosine
python main.py train \
    --config config/default.yaml \
    --optimizer adam \
    --scheduler step

# Disable scheduler (constant learning rate)
python main.py train \
    --config config/default.yaml \
    --scheduler none
```

**Important Note about Cosine Scheduler**: When using cosine annealing, ensure `T_max >= total_epochs`. The default config sets `T_max: null` which automatically uses the epochs value. If you're resuming training or extending epochs, make sure T_max is adjusted accordingly to avoid learning rate cycling that can cause loss degradation.

### Resume Training

Resume from a checkpoint:

```bash
python main.py train \
    --config config/model_x_sigma.yaml \
    --resume experiments/train/exp_20231215_120000/latest.pkl
```

### Training Output

Training creates an experiment directory:

```
experiments/train/exp_YYYYMMDD_HHMMSS/
├── config.yaml              # Configuration used
├── best_model.pkl           # Best model (lowest val loss)
├── latest.pkl               # Latest checkpoint
├── checkpoint_epoch_*.pkl   # Periodic checkpoints
├── train.log                # Detailed logs
├── metrics.json             # Training metrics
└── loss_curve.png           # Loss curves
```

## Testing Models

### Basic Testing

Test a trained model:

```bash
python main.py test \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pkl
```

### Test with Specific Configuration

```bash
python main.py test \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pkl \
    --config config/model_x_sigma.yaml \
    --test-samples 5000
```

### Test Output

Testing creates a test directory:

```
experiments/test/exp_YYYYMMDD_HHMMSS/
├── config.yaml              # Test configuration
├── test.log                 # Test logs
├── test_metrics.json        # Evaluation metrics
├── test_scatter_predictions.png    # Predictions vs targets (ω̂ scatter plot)
├── test_noise_distribution.png     # Noise level distribution
└── test_sample_images.png          # Sample clean/noisy images
```

### Training Output Plots

After training completes, the following additional plots are generated:

```
experiments/train/exp_YYYYMMDD_HHMMSS/
├── ...
├── train_scatter_predictions.png   # Training data: ω̂ predictions vs targets
├── val_scatter_predictions.png     # Validation data: ω̂ predictions vs targets
└── loss_curve.png                  # Training/validation loss curves
```

### Interpretation of Metrics

The evaluation metrics include:

- **MSE**: Mean Squared Error between predictions and targets
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination (1.0 is perfect)
- **Mean Relative Error**: Average |prediction - target| / |target|
- **Median Relative Error**: Median relative error (more robust to outliers)

### Understanding Scatter Plots

The scatter plots visualize model performance by comparing:

- **X-axis (Target)**: Ground truth ω̂_target = ||x - x̃||² / σ³
  - Where x is clean image, x̃ is noisy image, σ is noise level
- **Y-axis (Prediction)**: Model output ω̂ after applying output transformation

**What to look for:**
- Points should cluster around the diagonal line (y = x)
- R² score closer to 1.0 indicates better fit
- Systematic deviations from diagonal indicate bias
- Scatter around diagonal indicates variance in predictions

The evaluation formula ω̂_target = ||x - x̃||² / σ³ is mathematically equivalent to ||ε||² / σ where ε = (x̃ - x) / σ. This provides a consistent evaluation metric across all loss types after applying appropriate output transformations.

## Using Trained Models

### Loading a Model in Python

#### From .pkl File (Recommended)

```python
import torch

# Load the entire model
model = torch.load('path/to/model.pkl', map_location='cuda')
model.eval()

print("Model loaded successfully!")
```

#### From .pth File

```python
import torch
from src.models import create_model

# Create model architecture
model = create_model('omega_x_sigma')

# Load state dict
checkpoint = torch.load('path/to/checkpoint.pth', map_location='cuda')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Model loaded successfully!")
```

### Making Predictions

#### For omega(x, sigma) Model

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image = Image.open('path/to/image.png')
image_tensor = transform(image).unsqueeze(0).cuda()

# Add noise
sigma = torch.tensor([2.0]).cuda()  # Noise level
noise = torch.randn_like(image_tensor) * sigma.view(-1, 1, 1, 1)
noisy_image = image_tensor + noise

# Predict
with torch.no_grad():
    prediction = model(noisy_image, sigma)

print(f"Predicted noise-level score gradient: {prediction.item()}")
```

#### For omega(x) Model

```python
# For models without explicit sigma input
with torch.no_grad():
    prediction = model(noisy_image)

print(f"Prediction: {prediction.item()}")
```

### Batch Prediction

```python
import torch
from torch.utils.data import DataLoader
from src.datasets import get_cifar10_dataloaders

# Load data
_, _, test_loader = get_cifar10_dataloaders(batch_size=64)

# Predict on batches
model.eval()
predictions = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.cuda()
        
        # Generate random noise levels
        sigma = torch.rand(images.size(0)).cuda() * 10.0
        
        # Add noise
        noise = torch.randn_like(images) * sigma.view(-1, 1, 1, 1)
        noisy_images = images + noise
        
        # Predict
        output = model(noisy_images, sigma)
        predictions.append(output.cpu())

predictions = torch.cat(predictions)
print(f"Predicted on {len(predictions)} samples")
```

## Configuration Guide

### Configuration File Structure

```yaml
# Model configuration
model:
  type: "omega_x_sigma"  # or "omega_x"
  backbone: "resnet18"

# Noise generation
noise:
  sigma_min: 0.01        # Minimum noise level
  sigma_max: 10.0        # Maximum noise level
  strategy: "log_uniform"  # Sampling strategy
  num_samples: 100       # For select_batch strategy

# Training parameters
training:
  loss_type: "omega_hat"  # Loss function
  epochs: 100
  
  # Optimizer configuration
  optimizer: "adam"      # "adam" or "sgd"
  optimizer_params:
    betas: [0.9, 0.999]  # For Adam
    eps: 1.0e-8
    momentum: 0.9        # For SGD
    nesterov: true
  
  learning_rate: 0.001
  weight_decay: 0.0
  
  # LR scheduler
  scheduler: "cosine"    # "cosine", "step", or "none"
  scheduler_params:
    T_max: null          # Auto-set to epochs (IMPORTANT: must be >= epochs)
    step_size: 30        # For step scheduler
    gamma: 0.1
  
  early_stopping: true
  patience: 10

# Data configuration
data:
  batch_size: 128
  num_workers: 4
  train_val_split: [0.85, 0.10]
  test_split: 0.05
  data_root: "./data"

# Checkpoint settings
checkpoint:
  save_format: "pkl"     # "pkl" or "pth"
  save_best: true
  save_every: 10

# Logging
logging:
  log_interval: 100      # Log every N batches
  plot_interval: 1       # Plot every N epochs

# Device and seed
device: "cuda"
seed: 42
```

### Key Configuration Options

#### Model Types

- `omega_x`: Model takes only image input
- `omega_x_sigma`: Model takes image and noise level

#### Loss Functions

The framework supports three loss types, all tested and validated:

1. **`omega_hat`**: Original formulation - ||x - x̃||² / σ³
   - Direct estimation of omega
   - Baseline approach, suitable for most applications
   - No output transformation needed

2. **`omega_epsilon`**: Epsilon-based - ||ε||² / σ
   - More numerically stable for small σ values
   - Recommended when using small noise levels or log-uniform sampling
   - No output transformation needed

3. **`omega_chi_zscore`**: Chi-squared z-score based estimation
   - Model learns normalized z-score: (||ε||² - d) / √(2d)
   - Provides stable training with zero-mean, unit-variance targets
   - Output transformation applied during inference: ω = (output × √(2d) + d) / σ
   - May benefit from longer training (200 epochs)

**Important Notes**: 
- Loss types 1-2 directly output omega (no transformation)
- Loss type 9 applies output transformation during inference only (not during training)
- All three loss types estimate the same quantity: ∇_σ log[σ^d p(x, σ)]

For mathematical details and transformation formulas, see [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md)

#### Noise Sampling Strategies

- `uniform`: Uniform sampling in [sigma_min, sigma_max]
- `log_uniform`: Uniform in log-space (more samples at low sigma)
- `select_batch`: Sample from pre-filtered values

#### Optimizers

- `adam`: Adam optimizer (recommended for most cases)
  - Parameters: `betas`, `eps`
- `sgd`: Stochastic Gradient Descent
  - Parameters: `momentum`, `nesterov`

#### Learning Rate Schedulers

- `cosine`: Cosine annealing (recommended)
  - **IMPORTANT**: `T_max` should be >= total epochs to avoid LR cycling
  - Set `T_max: null` to auto-use epochs value
  - If T_max < epochs, LR will restart and increase, causing loss degradation
- `step`: Step decay at fixed intervals
  - Parameters: `step_size`, `gamma`
- `none`: No scheduler (constant learning rate)

## Advanced Usage

### Custom Loss Function

Add a custom loss in `src/training/loss_functions.py`:

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def forward(self, output, clean_images, noisy_images, sigma):
        # Your custom loss implementation
        target = # compute your target
        loss = # compute your loss
        return loss
```

Register it in `LossFactory`:

```python
def get_loss(loss_type='omega_hat', image_dim=3072):
    if loss_type == 'custom':
        return CustomLoss()
    # ... existing code
```

### Custom Noise Strategy

Add a custom strategy in `src/datasets/noise_generator.py`:

```python
def _custom_sampling(self, batch_size):
    """Your custom sampling strategy."""
    # Example: Sample from a specific distribution
    sigma = # your sampling logic
    return sigma
```

Update `generate_sigma` method:

```python
def generate_sigma(self, batch_size):
    if self.strategy == 'custom':
        return self._custom_sampling(batch_size)
    # ... existing code
```

### Exporting Models

#### Export to .pkl

```bash
python main.py export \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pth \
    --output my_omega_model.pkl
```

#### Export to .pth

```bash
python main.py export \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pkl \
    --output my_omega_model.pth
```

### Using Models in Other Projects

```python
# In your external project
import sys
sys.path.append('/path/to/sigma-score-estimator')

from src.models import create_model
import torch

# Load model
model = torch.load('path/to/my_omega_model.pkl')
model.eval()

# Use in your application
def estimate_noise_gradient(image, sigma):
    with torch.no_grad():
        return model(image, sigma)
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```bash
# Reduce batch size
python main.py train --config config/default.yaml --batch-size 64

# Use gradient accumulation (modify trainer.py)
# Reduce num_workers
python main.py train --config config/default.yaml --batch-size 128 --num-workers 2
```

#### Slow Training

```bash
# Enable mixed precision training (modify trainer.py to use torch.cuda.amp)
# Increase num_workers
python main.py train --config config/default.yaml --num-workers 8

# Use smaller model (if implementing other backbones)
```

#### Loss Not Decreasing

- Check learning rate (try 0.0001 - 0.001)
- Verify data normalization
- Try different loss functions
- Check noise sampling range (sigma_min, sigma_max)
- Increase batch size

#### Loss Increasing After Initial Decrease (LR Cycling Issue)

If your loss decreases initially but then starts increasing after a certain epoch (e.g., epoch 100), this is likely due to learning rate cycling in the cosine scheduler:

**Problem**: When `T_max < total_epochs`, the cosine scheduler completes a full cycle and the learning rate starts increasing again, causing loss degradation.

**Solution**:
1. Set `T_max: null` in your config to auto-use epochs value
2. Or manually set `T_max` >= your total training epochs
3. Check your training logs for the warning: "T_max is less than epochs"

Example fix in `config/default.yaml`:
```yaml
training:
  epochs: 200
  scheduler: "cosine"
  scheduler_params:
    T_max: null  # Will automatically use 200
```

#### Model Not Learning

- Verify data loading is correct
- Check if targets are computed correctly
- Try simpler model first (omega_x)
- Reduce noise range initially

### Debugging Tips

#### Enable Verbose Logging

Modify `src/utils/logger.py` to set level to `logging.DEBUG`

#### Visualize Training Data

```python
from src.datasets import get_cifar10_dataloaders
from src.utils import visualize_sample_images

train_loader, _, _ = get_cifar10_dataloaders()

images, _ = next(iter(train_loader))
# Visualize to ensure data is correct
```

#### Check Model Output Range

```python
model.eval()
with torch.no_grad():
    output = model(test_images, test_sigma)
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"Output mean: {output.mean():.3f}")
```

## Best Practices

1. **Start Small**: Begin with fewer epochs and smaller datasets to verify everything works
2. **Monitor Validation Loss**: Use early stopping to prevent overfitting
3. **Save Checkpoints**: Use save_every to save periodic checkpoints
4. **Experiment**: Try different loss functions and noise strategies
5. **Document**: Keep track of configurations and results
6. **Visualize**: Regularly check the plots to understand training progress
7. **Test Thoroughly**: Evaluate on held-out test set before deployment

## Performance Optimization

### Training Speed

- Use DataLoader with `num_workers > 0` and `pin_memory=True`
- Use mixed precision training (torch.cuda.amp)
- Increase batch size if memory allows
- Use multiple GPUs with DataParallel or DistributedDataParallel

### Model Quality

- Train for sufficient epochs (monitor validation loss)
- Use log_uniform noise sampling for better coverage
- Try ensemble of multiple models
- Use larger batch sizes for more stable gradients

## Further Reading

- [README.md](README.md): Project overview
- [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md): Mathematical derivation
- [config/](config/): Example configurations
- [src/](src/): Source code documentation

For questions or issues, please check the documentation or create an issue in the repository.

