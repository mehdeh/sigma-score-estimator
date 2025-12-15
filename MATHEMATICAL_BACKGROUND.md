# Mathematical Background

This document provides the mathematical foundation for training the $\omega_{\phi}(\mathbf{x}, \sigma)$ model to estimate the noise-level score gradient $\nabla_\sigma \log p(\mathbf{x}, \sigma)$.

## Table of Contents

1. [Overview](#overview)
2. [Problem Formulation](#problem-formulation)
3. [Relationship Between Denoiser and Score Function](#relationship-between-denoiser-and-score-function)
4. [Deriving the Noise-Level Score Gradient](#deriving-the-noise-level-score-gradient)
5. [Loss Function for Training](#loss-function-for-training)
6. [Implementation Details](#implementation-details)

## Overview

In diffusion models, we often need to estimate two gradients:
- $\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma)$: The data-space score function
- $\nabla_\sigma \log p(\mathbf{x}, \sigma)$: The noise-level score gradient

This repository focuses on estimating the noise-level score gradient using a neural network $\omega_{\phi}(\mathbf{x}, \sigma)$.

## Problem Formulation

Given a clean data sample $\tilde{\mathbf{x}} \sim p_{\text{data}}$, we add Gaussian noise with standard deviation $\sigma$ to obtain noisy data:

$$\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})$$

The joint distribution of noisy data is:

$$p(\mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \, \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \, d\tilde{\mathbf{x}}$$

Our goal is to estimate $\nabla_\sigma \log p(\mathbf{x}, \sigma)$.

## Relationship Between Denoiser and Score Function

### Optimal Denoising Function

The optimal denoising function $D^*(\mathbf{x}, \sigma)$ minimizes the expected MSE loss:

$$\mathcal{L}(D; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \lVert D(\tilde{\mathbf{x}} + \mathbf{z}, \sigma) - \tilde{\mathbf{x}} \rVert^2_2$$

Taking the gradient with respect to $D$ and setting it to zero yields the closed-form solution:

$$D^*(\mathbf{x}, \sigma) = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \, \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \, \tilde{\mathbf{x}} \, d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}$$

This is the conditional expectation: $D^*(\mathbf{x}, \sigma) = \mathbb{E}[\tilde{\mathbf{x}} | \mathbf{x}, \sigma]$

### Connection to Score Function

The data-space score function is related to the denoiser by:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = -\frac{1}{\sigma^2} \left[ \mathbf{x} - D(\mathbf{x}, \sigma) \right]$$

**Proof:** Starting from the definition of the score function and using properties of Gaussian distributions:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x}, \sigma)}{p(\mathbf{x}, \sigma)}$$

$$= \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \, \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[ -\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2} \right] d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}$$

$$= -\frac{1}{\sigma^2} \left[ \mathbf{x} - D(\mathbf{x}, \sigma) \right]$$

## Deriving the Noise-Level Score Gradient

### Gradient of Log-Probability w.r.t. Sigma

We need to compute $\nabla_\sigma \log p(\mathbf{x}, \sigma)$. Starting with the Gaussian PDF:

$$\log \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = -\frac{d}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2$$

Taking the gradient with respect to $\sigma$:

$$\nabla_\sigma \log \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = -\frac{d}{\sigma} + \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3}$$

### Closed-Form Expression

The noise-level score gradient has the closed form:

$$\nabla_\sigma \log p(\mathbf{x}, \sigma) = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \, \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[ \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} - \frac{d}{\sigma} \right] d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}$$

This can be rewritten as:

$$\nabla_\sigma \log p(\mathbf{x}, \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} | \mathbf{x}, \sigma} \left[ \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} - \frac{d}{\sigma} \right]$$

## Loss Function for Training

### Modified Estimator

We define a modified estimator $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ related to $\omega_\theta(\mathbf{x}, \sigma)$ by:

$$\hat{\omega}_\theta(\mathbf{x}, \sigma) = \omega_\theta(\mathbf{x}, \sigma) + \frac{d}{\sigma}$$

This modification simplifies the loss function.

### Derived Loss Function

Through detailed derivation (see chapter3.tex equations 559-567), the loss function for training $\omega_{\phi}(\mathbf{x}, \sigma)$ is:

$$\mathcal{L}(\hat{\omega}; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})} \left[ \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} \right)^2 \right]$$

This is the **primary loss function** used in this repository (loss_type: `omega_hat`).

### Loss Interpretation

The loss minimizes the squared difference between:
- **Model prediction**: $\hat{\omega}_\theta(\mathbf{x}, \sigma)$
- **Target**: $\frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3}$

Where:
- $\mathbf{x}$ is the noisy image
- $\tilde{\mathbf{x}}$ is the clean image
- $\sigma$ is the noise level
- $d$ is the data dimensionality (e.g., $d = 3072$ for CIFAR-10)

## Implementation Details

### Training Procedure

1. **Sample** a clean image $\tilde{\mathbf{x}} \sim p_{\text{data}}$
2. **Sample** a noise level $\sigma$ from the desired distribution
3. **Add noise**: $\mathbf{x} = \tilde{\mathbf{x}} + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$
4. **Forward pass**: Compute $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ using the neural network
5. **Compute target**: $\text{target} = \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3}$
6. **Compute loss**: $\mathcal{L} = (\hat{\omega}_\theta(\mathbf{x}, \sigma) - \text{target})^2$
7. **Backpropagate** and update model parameters

### Alternative Loss Functions

The repository also implements several alternative loss formulations for experimentation:

1. **Normalized Loss**:
   $$\mathcal{L} = \frac{(\omega_\theta(\mathbf{x}, \sigma) - \sigma)^2}{\sigma}$$

2. **Relative Loss**:
   $$\mathcal{L} = \left(\frac{\omega_\theta(\mathbf{x}, \sigma)}{\sigma} - 1\right)^2$$

3. **Sigma-Cal Loss**:
   $$\mathcal{L} = (\omega_\theta(\mathbf{x}, \sigma) - (\sigma_{\text{cal}} - \sigma))^2$$
   
   Where $\sigma_{\text{cal}}$ is the empirical standard deviation of the added noise.

### Noise Sampling Strategies

The distribution of $\sigma$ during training affects model performance:

- **Uniform**: $\sigma \sim \mathcal{U}(\sigma_{\min}, \sigma_{\max})$
- **Log-uniform**: $\log \sigma \sim \mathcal{U}(\log \sigma_{\min}, \log \sigma_{\max})$
- **Pre-filtered batch**: Sample from a pre-defined set of values

## Key Equations Summary

| Concept | Equation |
|---------|----------|
| Score function | $\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = -\frac{1}{\sigma^2} [\mathbf{x} - D(\mathbf{x}, \sigma)]$ |
| Noise-level score | $\nabla_\sigma \log p(\mathbf{x}, \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \| \mathbf{x}, \sigma} \left[ \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} - \frac{d}{\sigma} \right]$ |
| Modified estimator | $\hat{\omega}_\theta(\mathbf{x}, \sigma) = \omega_\theta(\mathbf{x}, \sigma) + \frac{d}{\sigma}$ |
| Training loss | $\mathcal{L} = \mathbb{E} \left[ \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} \right)^2 \right]$ |

## References

For the complete mathematical derivation with all intermediate steps, see `sigma_model/chapter3.tex` lines 189-654.

## Notes

- The dimension $d = C \times H \times W$ where $C$ is the number of channels, $H$ is height, and $W$ is width
- For CIFAR-10: $d = 3 \times 32 \times 32 = 3072$
- The loss function is derived to be an unbiased estimator of the true gradient
- Two model variants are supported:
  - $\omega_{\phi}(\mathbf{x})$: Noise level implicit in training
  - $\omega_{\phi}(\mathbf{x}, \sigma)$: Noise level as explicit input

