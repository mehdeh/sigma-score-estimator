# Sigma-Score-Estimator

A modular PyTorch framework for training ResNet-based models to estimate the **noise-level score gradient** $\nabla_\sigma \log p(\mathbf{x}, \sigma)$ or its corrected form $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$, which are fundamental components in diffusion models for adaptive noise scheduling during sampling.

## 📋 Overview

This repository implements deep learning models that estimate the gradient of log-probability with respect to the noise level $\sigma$. The framework provides:

- **Theoretical Foundation**: Complete mathematical derivation connecting denoising models to score functions
- **Two Estimator Types**: 
  - $\omega_{\phi}(\mathbf{x}, \sigma)$ for estimating $\nabla_\sigma \log p(\mathbf{x}, \sigma)$
  - $\hat{\omega}_{\phi}(\mathbf{x}, \sigma)$ for estimating $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$ (corrected objective)
- **Practical Implementation**: Multiple loss formulations enabling better control over noise scheduling in diffusion models

### Key Features

- **Two Model Variants:**
  - $\omega_{\phi}(\mathbf{x})$: Image-only input
  - $\omega_{\phi}(\mathbf{x}, \sigma)$: Image and noise level input

- **Multiple Loss Functions:**
  - 9 different loss formulations with rigorous theoretical foundation
  - Omega-based losses (Types 1-4, 9): Direct estimation of corrected sigma score $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$
  - Sigma-based losses (Types 5-8): Indirect estimation via noise level prediction
  - All formulations derived from score matching principles
  - Output transformations applied during inference (not training) for types 5-9
  - See [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md) for complete derivations
  
- **Flexible Noise Sampling:**
  - Uniform sampling
  - Log-uniform sampling
  - Pre-filtered batch selection

- **Complete Training Pipeline:**
  - Modular architecture inspired by modern ML practices
  - Comprehensive logging and visualization
  - Support for both `.pkl` and `.pth` checkpoint formats
  - Early stopping and learning rate scheduling

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
cd /path/to/sigma-score-estimator

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Train with default configuration
python main.py train --config config/default.yaml

# Train omega(x, sigma) model with custom settings
python main.py train \
    --config config/model_x_sigma.yaml \
    --epochs 100 \
    --learning-rate 0.001 \
    --batch-size 128 \
    --early-stopping true

# Train omega(x) model (image-only)
python main.py train --config config/model_x.yaml
```

### Testing a Trained Model

```bash
# Test a trained model
python main.py test \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pkl

# Test with specific number of samples
python main.py test \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pkl \
    --test-samples 1000
```

### Exporting a Model

```bash
# Export model to .pkl format
python main.py export \
    --checkpoint experiments/train/exp_20231215_120000/best_model.pth \
    --output my_omega_model.pkl
```

## 📁 Project Structure

```
sigma-score-estimator/
├── config/                      # Configuration files
│   ├── default.yaml            # Default configuration
│   ├── model_x.yaml            # Config for omega(x)
│   └── model_x_sigma.yaml      # Config for omega(x,sigma)
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   ├── data/                   # Data loading and noise generation
│   ├── training/               # Training and evaluation
│   └── utils/                  # Utilities (logging, checkpointing, etc.)
├── experiments/                # Experiment outputs (auto-generated)
│   ├── train/                  # Training runs (exp_YYYYMMDD_HHMMSS)
│   └── test/                   # Test runs (exp_YYYYMMDD_HHMMSS)
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── MATHEMATICAL_BACKGROUND.md  # Mathematical derivation
└── USAGE_GUIDE.md             # Detailed usage guide
```

## 🔬 Mathematical Background

### Core Theory

The framework builds on the fundamental relationship between **denoising models** and **score functions**:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = -\frac{1}{\sigma^2}[\mathbf{x} - D(\mathbf{x}, \sigma)]$$

where $D(\mathbf{x}, \sigma)$ is a pretrained denoising model. This allows using existing denoising models to compute score functions.

### Sigma Score Estimation

For the noise-level score gradient, we derive the closed-form expression:

$$\nabla_\sigma \log p(\mathbf{x}, \sigma) = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right] d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}$$

The model $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ is trained to estimate the **corrected gradient** $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$ using:

$$\mathcal{L}(\hat{\omega}; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})} \left[ \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{ \lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2 }{\sigma^3} \right)^2 \right]$$

The corrected objective ensures monotonic decrease of $\sigma$ during sampling, enabling better mode coverage in multimodal distributions.

**Complete derivation**: See [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md) for:
- Relationship between denoising models and score functions
- Derivation of sigma score closed-form expression
- Loss function derivation
- Corrected objective function ($\sigma^d p(\mathbf{x}, \sigma)$)
- Eight practical loss formulations

## 📊 Configuration

All training parameters can be configured via YAML files or command-line arguments. Key configuration sections:

- **Model**: Architecture type (omega_x or omega_x_sigma)
- **Noise**: Noise level range and sampling strategy
- **Training**: Loss function, learning rate, epochs, etc.
- **Data**: Batch size, data splits, augmentation
- **Checkpoint**: Save format and frequency

Example configuration:

```yaml
model:
  type: "omega_x_sigma"
  
noise:
  sigma_min: 0.00001  # 1e-5
  sigma_max: 4.0
  strategy: "uniform"
  
training:
  loss_type: "omega_hat"  # Options: omega_hat, omega_epsilon, omega_chi_approx,
                          #          omega_chi_mean, sigma_direct, sigma_normalized,
                          #          sigma_relative, sigma_calibrated, omega_chi_zscore
  epochs: 100
  learning_rate: 0.001
  sigma_cal: 0.0          # Calibration parameter (for sigma_calibrated loss)
```

## 📈 Monitoring Training

Training progress is automatically saved to the experiments directory (flat files per run):

```
experiments/train/exp_YYYYMMDD_HHMMSS/
├── config.yaml              # Configuration used
├── best_model.pkl           # Best model (lowest val loss)
├── latest.pkl               # Latest checkpoint
├── checkpoint_epoch_*.pkl   # Periodic checkpoints
├── train.log                # Training logs
├── metrics.json             # Training metrics
└── loss_curve.png           # Loss curves
```

## 🔍 Using Trained Models

### Loading a Model

```python
import torch
from src.models import create_model

# Load full model from .pkl
model = torch.load('path/to/model.pkl')

# Or load with architecture
model = create_model('omega_x_sigma')
checkpoint = torch.load('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
```

### Making Predictions

```python
# For omega(x, sigma) model
output = model(noisy_images, sigma_values)

# For omega(x) model
output = model(noisy_images)
```

## 🛠️ Advanced Usage

### Custom Loss Functions

Add new loss functions in `src/training/loss_functions.py`:

```python
class CustomLoss(nn.Module):
    def forward(self, output, clean_images, noisy_images, sigma):
        # Your loss implementation
        pass
```

### Custom Noise Strategies

Add new sampling strategies in `src/data/noise_generator.py`:

```python
def _custom_sampling(self, batch_size):
    # Your sampling implementation
    pass
```

## 📚 Documentation

- [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md): Complete mathematical derivation
- [USAGE_GUIDE.md](USAGE_GUIDE.md): Detailed usage instructions and examples

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code is well-documented in English
- Tests pass
- Code style is consistent with the project

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{sigma_score_estimator,
  title={Sigma-Score-Estimator: Training Models for Noise-Level Score Gradient Estimation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sigma-score-estimator}
}
```

## 📄 License

This project is free and open source and may be used, modified, and distributed for research, educational, and commercial purposes without restriction.

## 🙏 Acknowledgments

- Inspired by the [`cifar-noise-estimation` repository](https://github.com/mehdeh/cifar-noise-estimation)
- Built with PyTorch and modern ML best practices
- Mathematical formulation based on:
  - Score-based generative modeling [Song et al., 2021]
  - Denoising diffusion probabilistic models [Ho et al., 2020]
  - Connection between denoising and score matching [Karras et al., 2022; Vincent, 2011]
  - Theory of score matching [Hyvärinen & Dayan, 2005]

---

For detailed usage instructions, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

For mathematical background, see [MATHEMATICAL_BACKGROUND.md](MATHEMATICAL_BACKGROUND.md).

