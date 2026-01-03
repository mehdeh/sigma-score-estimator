# Mathematical Background: Sigma Score Estimation for Diffusion Models

## Overview

This document provides the complete mathematical foundation for estimating the **noise-level score gradient** $\nabla_\sigma \log p(\mathbf{x}, \sigma)$ in diffusion models. We derive the theoretical basis for training neural networks to estimate this gradient, present multiple loss function formulations, and explain the relationship between denoising models and score functions.

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Relationship Between Denoising and Score Functions](#relationship-between-denoising-and-score-functions)
3. [Deriving the Sigma Score Gradient](#deriving-the-sigma-score-gradient)
4. [Loss Function for Training Sigma Score Estimator](#loss-function-for-training-sigma-score-estimator)
5. [Corrected Objective Function](#corrected-objective-function)
6. [Loss Function Formulations](#loss-function-formulations)
7. [Implementation Notes](#implementation-notes)
8. [Model Evaluation](#8-model-evaluation)
9. [References](#references)

---

## 1. Theoretical Foundation

### 1.1 The Forward Noising Process

Consider clean data $\tilde{\mathbf{x}} \sim p_{\text{data}}(\tilde{\mathbf{x}})$ and the forward noising process:

$$
\mathbf{x} = \tilde{\mathbf{x}} + \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})
$$

This defines the joint distribution:

$$
p(\mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) d\tilde{\mathbf{x}}
$$

### 1.2 The Score Function

The **score function** with respect to $\mathbf{x}$ is defined as:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x}, \sigma)}{p(\mathbf{x}, \sigma)}
$$

Similarly, the **sigma score** (score with respect to $\sigma$) is:

$$
\nabla_\sigma \log p(\mathbf{x}, \sigma) = \frac{\nabla_\sigma p(\mathbf{x}, \sigma)}{p(\mathbf{x}, \sigma)}
$$

---

## 2. Relationship Between Denoising and Score Functions

### 2.1 The Denoising Model

A **denoising model** $D(\mathbf{x}, \sigma)$ is trained to predict the clean image $\tilde{\mathbf{x}}$ from noisy observation $\mathbf{x}$ using the loss function:

$$
\mathcal{L}(D; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \lVert D(\tilde{\mathbf{x}} + \mathbf{z}, \sigma) - \tilde{\mathbf{x}} \rVert^2_2
$$

This can be rewritten as:

$$
\mathcal{L}(D; \sigma) = \int \mathcal{L}(D; \mathbf{x}, \sigma) d\mathbf{x}
$$

where

$$
\mathcal{L}(D; \mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \lVert D(\mathbf{x}, \sigma) - \tilde{\mathbf{x}} \rVert_2^2 d\tilde{\mathbf{x}}
$$

### 2.2 Optimal Denoiser

The optimal denoiser is found by minimizing $\mathcal{L}(D; \mathbf{x}, \sigma)$ for each fixed $\mathbf{x}$:

$$
\nabla_D \mathcal{L}(D; \mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \cdot 2[D(\mathbf{x}, \sigma) - \tilde{\mathbf{x}}] d\tilde{\mathbf{x}} = 0
$$

Solving this equation yields the closed-form solution:

$$
D^*(\mathbf{x}, \sigma) = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \tilde{\mathbf{x}} d\tilde{\mathbf{x}}}{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) d\tilde{\mathbf{x}}} = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \tilde{\mathbf{x}} d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}
$$

This is the **conditional expectation** of clean data given noisy observation.

### 2.3 Connecting Denoiser to Score Function

Using the properties of Gaussian distributions:

$$
\nabla_{\mathbf{x}} \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \cdot \left[-\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}\right]
$$

We can compute:

$$
\nabla_{\mathbf{x}} p(\mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[-\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}\right] d\tilde{\mathbf{x}}
$$

Therefore, the score function is:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = \frac{\nabla_{\mathbf{x}} p(\mathbf{x}, \sigma)}{p(\mathbf{x}, \sigma)} = -\frac{1}{\sigma^2}\left[\mathbf{x} - \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \tilde{\mathbf{x}} d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}\right]
$$

**Key Result**: Using the optimal denoiser $D^*(\mathbf{x}, \sigma)$, we obtain:

$$
\boxed{\nabla_{\mathbf{x}} \log p(\mathbf{x}, \sigma) = -\frac{1}{\sigma^2}[\mathbf{x} - D(\mathbf{x}, \sigma)]}
$$

This fundamental relationship shows that a pretrained denoising model can be used to compute the score function [1, 2].

---

## 3. Deriving the Sigma Score Gradient

### 3.1 Computing the Gradient with Respect to Sigma

Starting from the Gaussian log-density:

$$
\log \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2
$$

Taking the derivative with respect to $\sigma$:

$$
\nabla_\sigma \log \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = -\frac{d}{\sigma} + \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}
$$

Using the chain rule:

$$
\nabla_\sigma \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) = \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \cdot \left[\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right]
$$

### 3.2 Sigma Score Closed Form

Computing the gradient of $p(\mathbf{x}, \sigma)$ with respect to $\sigma$:

$$
\nabla_\sigma p(\mathbf{x}, \sigma) = \int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right] d\tilde{\mathbf{x}}
$$

Therefore:

$$
\boxed{\nabla_\sigma \log p(\mathbf{x}, \sigma) = \frac{\int p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right] d\tilde{\mathbf{x}}}{p(\mathbf{x}, \sigma)}}
$$

where $d$ is the dimensionality of $\mathbf{x}$ (e.g., $d = 3 \times 32 \times 32 = 3072$ for CIFAR-10).

---

## 4. Loss Function for Training Sigma Score Estimator

### 4.1 Initial Loss Formulation

To train a model $\omega_\theta(\mathbf{x}, \sigma)$ to estimate $\nabla_\sigma \log p(\mathbf{x}, \sigma)$, we use:

$$
\mathcal{L}(\omega; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})} \left[\nabla_\sigma \log p(\mathbf{x}, \sigma) - \omega_\theta(\mathbf{x}, \sigma)\right]^2
$$

### 4.2 Deriving the Tractable Loss

Expanding the squared term:

$$
\mathcal{L}(\omega; \sigma) = \mathbb{E}\left[[\nabla_\sigma \log p]^2\right] + \mathbb{E}\left[\omega_\theta^2\right] - 2\mathbb{E}\left[\nabla_\sigma \log p \cdot \omega_\theta\right]
$$

The cross term can be simplified using integration by parts:

$$
\begin{align}
\mathbb{E}_{\tilde{\mathbf{x}}} \mathbb{E}_{\mathbf{x}|\tilde{\mathbf{x}}}\left[\nabla_\sigma \log p(\mathbf{x}, \sigma) \omega_\theta(\mathbf{x}, \sigma)\right] &= \int \nabla_\sigma p(\mathbf{x}, \sigma) \omega_\theta(\mathbf{x}, \sigma) d\mathbf{x}\\
&= \iint p_{\text{data}}(\tilde{\mathbf{x}}) \mathcal{N}(\mathbf{x}; \tilde{\mathbf{x}}, \sigma^2 \mathbf{I}) \left[\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right] \omega_\theta(\mathbf{x}, \sigma) d\tilde{\mathbf{x}} d\mathbf{x}
\end{align}
$$

This yields:

$$
\mathbb{E}_{\tilde{\mathbf{x}}} \mathbb{E}_{\mathbf{x}|\tilde{\mathbf{x}}}\left[\nabla_\sigma \log p \cdot \omega_\theta\right] = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})}\left[\left(\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right) \omega_\theta(\mathbf{x}, \sigma)\right]
$$

**Important**: This expression **does not depend on** $p(\mathbf{x}, \sigma)$, making it tractable!

### 4.3 Final Loss Function

Dropping the constant term $\mathbb{E}[[\nabla_\sigma \log p]^2]$ (which doesn't affect optimization), we get:

$$
\boxed{\mathcal{L}(\omega; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})} \left[\omega_\theta(\mathbf{x}, \sigma) - \left(\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}\right)\right]^2}
$$

This is the **fundamental training loss** for estimating $\omega_\theta(\mathbf{x}, \sigma) \approx \nabla_\sigma \log p(\mathbf{x}, \sigma)$.

---

## 5. Corrected Objective Function

### 5.1 Motivation for Correction

The gradient $\nabla_\sigma \log p(\mathbf{x}, \sigma)$ contains the term:

$$
\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} - \frac{d}{\sigma}
$$

This term can be **negative** in certain regions, causing $\sigma$ to increase rather than decrease during optimization. This prevents reaching all modes of the distribution [3].

### 5.2 Modified Objective

To ensure monotonic decrease of $\sigma$, we modify the objective function by multiplying $p(\mathbf{x}, \sigma)$ by $\sigma^d$:

$$
\text{Optimize: } \sigma^d p(\mathbf{x}, \sigma) \text{ instead of } p(\mathbf{x}, \sigma)
$$

**Justification**: The Gaussian normalization factor is $(2\pi\sigma^2)^{-d/2}$, so multiplying by $\sigma^d$ removes this normalization dependency.

### 5.3 Modified Sigma Score

The gradient of the modified objective is:

$$
\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)] = \nabla_\sigma[\log(\sigma^d) + \log p(\mathbf{x}, \sigma)] = \frac{d}{\sigma} + \nabla_\sigma \log p(\mathbf{x}, \sigma)
$$

### 5.4 Modified Estimator and Loss

We define a new estimator $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ for the modified gradient:

$$
\hat{\omega}_\theta(\mathbf{x}, \sigma) = \omega_\theta(\mathbf{x}, \sigma) + \frac{d}{\sigma}
$$

The corresponding loss function becomes:

$$
\boxed{\mathcal{L}(\hat{\omega}; \sigma) = \mathbb{E}_{\tilde{\mathbf{x}} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})} \left[\hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}\right]^2}
$$

**Key distinction**:
- $\omega_\theta(\mathbf{x}, \sigma)$ estimates $\nabla_\sigma \log p(\mathbf{x}, \sigma)$ 

- $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ estimates $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$

The modified loss ensures that $\sigma$ decreases monotonically, enabling better mode coverage in multimodal distributions.

---

## 6. Loss Function Formulations

Building on the theoretical foundation, we present eight practical loss function formulations for training neural networks.

### 6.1 Core Definitions

Given clean data $\tilde{\mathbf{x}}$ and noisy observation $\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})$:

$$
\mathbf{x} = \tilde{\mathbf{x}} + \sigma\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

Therefore:

$$
\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2 = \sigma^2 \lVert\epsilon\rVert_2^2
$$

### 6.2 Chi-Squared Distribution Properties

Since $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ with dimension $d$:

$$
\lVert\epsilon\rVert_2^2 \sim \chi^2_d
$$

**Key properties:**
- Expectation: $\mathbb{E}[\lVert\epsilon\rVert_2^2] = d$
- Variance: $\text{Var}[\lVert\epsilon\rVert_2^2] = 2d$
- Normal approximation: $\lVert\epsilon\rVert_2^2 \approx d + \sqrt{2d} \cdot Z$, where $Z \sim \mathcal{N}(0, 1)$

---

### Loss Type 1: `omega_hat` (Corrected Formulation)

**Description**: Direct estimation using the corrected objective (Section 5).

**Target**:

$$
\hat{\omega}_{\text{target}} = \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}
$$

**Loss Function**:

$$
\mathcal{L}_1 = \left(\hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}\right)^2
$$

**Note**: This corresponds to estimating $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$.

---

### Loss Type 2: `omega_epsilon` (Epsilon-Based Formulation)

**Description**: Estimation using the noise $\epsilon$ directly.

**Target**: 

$$
\hat{\omega}_{\text{target}} = \frac{\lVert\epsilon\rVert_2^2}{\sigma}
$$

**Loss Function**:

$$
\mathcal{L}_2 = \left(\hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert\epsilon\rVert_2^2}{\sigma}\right)^2
$$

**Advantage**: More numerically stable when $\sigma$ is small.

---

### Loss Type 9: `omega_chi_zscore` (Chi-Squared Z-Score)

**Description**: Model learns the z-score (standardized value) of the chi-squared distribution.

**Target**: 

$$
\acute{\omega}_{\text{target}} = \frac{\lVert\epsilon\rVert_2^2 - d}{\sqrt{2d}}
$$

**Loss Function**:

$$
\mathcal{L}_9 = \left(\acute{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert\epsilon\rVert_2^2 - d}{\sqrt{2d}}\right)^2
$$

**Output Transformation** (applied after training):

$$
\hat{\omega} = \frac{\acute{\omega}_\theta \cdot \sqrt{2d} + d}{\sigma}
$$

**Advantage**: Learns a normalized quantity (z-score) with approximately zero mean and unit variance, which may provide more stable training dynamics. Combines the benefits of chi-squared distribution knowledge with direct relationship to sigma.

**Mathematical Justification**: 

Since $\lVert\epsilon\rVert_2^2 \sim \chi^2_d$ with $\mathbb{E}[\lVert\epsilon\rVert_2^2] = d$ and $\text{Var}[\lVert\epsilon\rVert_2^2] = 2d$, the z-score standardization yields:

$$
Z = \frac{\lVert\epsilon\rVert_2^2 - d}{\sqrt{2d}} \xrightarrow{d \to \infty} \mathcal{N}(0, 1)
$$

By the Central Limit Theorem, this approximation becomes increasingly accurate for high-dimensional data (e.g., $d = 3072$ for CIFAR-10).

---

### Summary Table

| Method Type | Name | Target Quantity | Key Feature | Estimates |
|-------------|------|----------------|-------------|-----------|
| **Trained Models** | | | | |
| Loss Type 1 | `omega_hat` | $\frac{\\|\mathbf{x} - \tilde{\mathbf{x}}\\|^2}{\sigma^3}$ | Corrected formulation | $\nabla_\sigma \log[\sigma^d p]$ |
| Loss Type 2 | `omega_epsilon` | $\frac{\\|\epsilon\\|^2}{\sigma}$ | Noise-based, numerically stable | $\nabla_\sigma \log[\sigma^d p]$ |
| Loss Type 9 | `omega_chi_zscore` | $\frac{\\|\epsilon\\|^2 - d}{\sqrt{2d}}$ | Chi-squared z-score | $\nabla_\sigma \log[\sigma^d p]$ |
| **Computational** | | | | |
| No training | `omega_edm` | $\frac{\\|\mathbf{x} - D_{EDM}(\mathbf{x}, \sigma)\\|^2}{\sigma^3}$ | Pretrained EDM denoiser | $\nabla_\sigma \log[\sigma^d p]$ |
| No training | `omega_expected` | $\frac{d}{\sigma}$ | Statistical expectation | $\mathbb{E}[\nabla_\sigma \log[\sigma^d p]]$ |

---

## Computational Method: EDM-Based Estimation

### Overview

The `omega_edm` method provides a training-free approach to estimate $\hat{\omega}$ by leveraging a pretrained EDM (Elucidating the Design Space of Diffusion Models) denoiser.

### Mathematical Formulation

**Target (from Loss Type 1)**:

$$
\hat{\omega}_{\text{target}} = \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}
$$

where $\tilde{\mathbf{x}}$ is the clean image.

**Computational Approach**:

Since we don't have access to the true clean image $\tilde{\mathbf{x}}$ during evaluation, we use the EDM denoiser to approximate it:

$$
\tilde{\mathbf{x}} \approx D_{\text{EDM}}(\mathbf{x}, \sigma)
$$

Thus, the computational estimate is:

$$
\hat{\omega}_{\text{edm}}(\mathbf{x}, \sigma) = \frac{\lVert\mathbf{x} - D_{\text{EDM}}(\mathbf{x}, \sigma)\rVert_2^2}{\sigma^3}
$$

### Relationship to Trained Methods

**Trained Methods**:
- Learn to predict $\hat{\omega}$ using clean-noisy image pairs during training
- Model: $\hat{\omega}_\theta(\mathbf{x}, \sigma) \approx \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}_{\text{true}}\rVert_2^2}{\sigma^3}$

**EDM Computational Method**:
- Uses pretrained denoiser to estimate $\tilde{\mathbf{x}}$
- Computes $\hat{\omega}$ directly from the formula
- No training phase required

### Advantages and Limitations

**Advantages**:
1. No training required (saves hours of computation)
2. Based on state-of-the-art pretrained EDM models
3. Mathematically grounded in the same target formula
4. Can serve as baseline for comparison

**Limitations**:
1. Quality depends on EDM denoiser performance
2. Cannot adapt to specific data distributions through training
3. Fixed capacity (cannot improve beyond EDM)
4. Requires downloading pretrained model (~200MB)

---

## Computational Method: Expected Value Estimation

### Overview

The `omega_expected` method provides the simplest computational approach by using the statistical expectation of the squared norm of Gaussian noise. This method requires no training and no pretrained models—only knowledge of the image dimensionality.

### Mathematical Formulation

**Target (from Loss Type 2)**:

$$
\hat{\omega}_{\text{target}} = \frac{\|\epsilon\|^2}{\sigma}
$$

where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ with dimension $d$.

**Statistical Property**:

Since $\epsilon$ is a standard Gaussian random vector with dimension $d$:

$$
\|\epsilon\|^2 \sim \chi^2_d
$$

with the following properties:
- Expectation: $\mathbb{E}[\|\epsilon\|^2] = d$
- Variance: $\text{Var}[\|\epsilon\|^2] = 2d$

**Computational Estimate**:

Using the expectation:

$$
\hat{\omega}_{\text{expected}}(\sigma) = \frac{\mathbb{E}[\|\epsilon\|^2]}{\sigma} = \frac{d}{\sigma}
$$

This provides the expected (mean) value of $\hat{\omega}$ for any noisy image with noise level $\sigma$.

### Interpretation

**What does this estimate?**

This method estimates the **expected value** of omega_hat:

$$
\mathbb{E}_{\epsilon \sim \mathcal{N}(0, \mathbf{I})}[\hat{\omega}] = \mathbb{E}\left[\frac{\|\epsilon\|^2}{\sigma}\right] = \frac{d}{\sigma}
$$

**Note**: This is a constant for a given noise level and image dimensionality. It does not depend on the actual image content, only on the statistical properties of the noise.

### Relationship to Other Methods

**Trained Methods**:
- Learn image-dependent predictions: $\hat{\omega}_\theta(\mathbf{x}, \sigma)$
- Adapt to specific data distributions
- Output varies with image content

**EDM Computational Method**:
- Uses pretrained denoiser: $\hat{\omega}_{\text{edm}}(\mathbf{x}, \sigma) = \frac{\|\mathbf{x} - D_{\text{EDM}}(\mathbf{x}, \sigma)\|^2}{\sigma^3}$
- Image-dependent (varies with $\mathbf{x}$)
- Requires model inference

**Expected Value Method**:
- Uses statistical expectation: $\hat{\omega}_{\text{expected}}(\sigma) = \frac{d}{\sigma}$
- Image-independent (only depends on $\sigma$)
- No model inference required
- Provides theoretical baseline

### Advantages and Limitations

**Advantages**:
1. **Simplest possible estimator**: No training, no models, just a formula
2. **Extremely fast**: No neural network inference required
3. **Mathematically grounded**: Based on well-known chi-squared distribution properties
4. **Perfect theoretical baseline**: Represents the expected value that trained models should approximate on average
5. **Zero setup**: No pretrained models to download

**Limitations**:
1. **Image-independent**: Cannot adapt to specific image content
2. **Only provides expectation**: Cannot capture variance or image-specific deviations
3. **Baseline only**: Not suitable for actual sampling (use trained models or EDM method instead)

**When to use**:
- As a sanity check for trained models
- For theoretical comparisons and analysis
- To verify that trained models are learning meaningful patterns beyond the baseline
- For quick approximate estimates when high accuracy is not critical

### Variance Analysis

While this method provides the expected value $\mathbb{E}[\hat{\omega}] = \frac{d}{\sigma}$, the actual values have variance:

$$
\text{Var}[\hat{\omega}] = \text{Var}\left[\frac{\|\epsilon\|^2}{\sigma}\right] = \frac{\text{Var}[\|\epsilon\|^2]}{\sigma^2} = \frac{2d}{\sigma^2}
$$

This means:
- For large $\sigma$: Low variance, expected value is a good approximation
- For small $\sigma$: High variance, individual values can deviate significantly from $\frac{d}{\sigma}$

**Standard deviation**: $\text{SD}[\hat{\omega}] = \frac{\sqrt{2d}}{\sigma}$

### Comparison with Ground Truth

For a specific noisy image with known clean image $\tilde{\mathbf{x}}$:

**Ground truth**: 
$$
\hat{\omega}_{\text{true}} = \frac{\|\mathbf{x} - \tilde{\mathbf{x}}\|^2}{\sigma^3} = \frac{\|\sigma \epsilon\|^2}{\sigma^3} = \frac{\|\epsilon\|^2}{\sigma}
$$

**Expected value estimate**:
$$
\hat{\omega}_{\text{expected}} = \frac{d}{\sigma}
$$

**Relationship**:
$$
\mathbb{E}[\hat{\omega}_{\text{true}}] = \hat{\omega}_{\text{expected}}
$$

On average over many samples, the expected value method provides the correct mean, but individual predictions will differ by:

$$
\hat{\omega}_{\text{true}} - \hat{\omega}_{\text{expected}} = \frac{\|\epsilon\|^2 - d}{\sigma}
$$

This difference is a zero-mean random variable with variance $\frac{2d}{\sigma^2}$.

---

## 7. Implementation Notes

### 7.1 Model Output Interpretation

**For Loss Types 1-2**: The model directly outputs $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ which estimates $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$.
- No transformation needed during inference
- Output is immediately usable for sampling
- Loss Type 2 (`omega_epsilon`) is more numerically stable for small $\sigma$ values

**For Loss Type 9**: The model outputs z-score of chi-squared distribution.
- During training: Model learns z-score $(||\epsilon||^2 - d) / \sqrt{2d}$
- During inference: Transform applied: $\hat{\omega} = (\acute{\omega} \cdot \sqrt{2d} + d) / \sigma$
- **Important**: Transformation is NOT part of loss computation
- Provides normalized learning target with approximately zero mean and unit variance

### 7.2 Converting Between Estimators

If $\omega_\theta$ estimates $\nabla_\sigma \log p(\mathbf{x}, \sigma)$, the corrected estimator is:

$$
\hat{\omega}_\theta(\mathbf{x}, \sigma) = \omega_\theta(\mathbf{x}, \sigma) + \frac{d}{\sigma}
$$

Conversely:

$$
\omega_\theta(\mathbf{x}, \sigma) = \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{d}{\sigma}
$$

### 7.3 Numerical Stability

- **Loss Type 1 (`omega_hat`)**: Stable for most $\sigma$ ranges, but may have numerical issues for very small $\sigma$ due to division by $\sigma^3$
- **Loss Type 2 (`omega_epsilon`)**: More numerically stable for small $\sigma$ values, recommended when using small noise levels
- **Loss Type 9 (`omega_chi_zscore`)**: Learns normalized quantities, provides stable training dynamics
- For very small $\sigma$ values, consider using log-space computations

### 7.4 Training Considerations

- **Loss Type 1 (`omega_hat`)**: Baseline formulation, suitable for most applications with moderate $\sigma$ ranges
- **Loss Type 2 (`omega_epsilon`)**: Recommended for small $\sigma$ values or when using log-uniform noise sampling
- **Loss Type 9 (`omega_chi_zscore`)**: Learns normalized z-score with zero mean and unit variance, potentially more stable training dynamics
- All three loss types have been tested and show good convergence with default hyperparameters
- The corrected objective (using $\hat{\omega}_\theta$) ensures monotonic decrease of $\sigma$ during sampling

### 7.5 Output Transformation Architecture

The output transformation is cleanly separated from loss computation:

**During Training**: 
- Loss functions compute MSE between model raw output and training target
- No transformation applied inside the loss function
- Models learn directly interpretable quantities (e.g., sigma, z-score)

**During Inference**: 
- Output transform converts model output to $\hat{\omega}_\theta$ for sampling
- Transform is applied automatically in the `OmegaEvaluator` class
- Ensures consistent evaluation across all loss types

**During Evaluation**: 
- Transform is applied before computing metrics
- All metrics are computed in $\hat{\omega}$ space
- Ground truth: $\hat{\omega}_{\text{true}} = ||\epsilon||^2 / \sigma$

This separation ensures:
1. **Simplicity**: Loss functions are mathematically simple and numerically stable
2. **Interpretability**: Model learns directly meaningful quantities
3. **Consistency**: Centralized transformation logic, easy to maintain and extend
4. **Correctness**: Transformation applied at the right time (inference, not training)

**Transformation Summary**:

| Loss Types | Model Learns | Training Target | Inference Transform | Final Output |
|------------|--------------|-----------------|---------------------|--------------|
| 1-2 | $\hat{\omega}$ | $\hat{\omega}$-related | None (identity) | $\hat{\omega}$ |
| 9 | z-score | $(\\|\epsilon\\|^2 - d) / \sqrt{2d}$ | $(\acute{\omega} \sqrt{2d} + d) / \sigma$ | $\hat{\omega}$ |

---

## 8. References

[1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *International Conference on Learning Representations (ICLR)*.

[2] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 26565-26577.

[3] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[4] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6840-6851.

[5] Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*, 23(7), 1661-1674.

[6] Hyvärinen, A., & Dayan, P. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. *Journal of Machine Learning Research*, 6, 695-709.

---

## 8. Model Evaluation

### 8.1 Evaluation Formula

All models are evaluated using a consistent metric in $\hat{\omega}$ space, regardless of which loss function was used during training. The ground truth target for evaluation is computed as:

$$
\hat{\omega}_{\text{target}} = \frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3}
$$

where:
- $\mathbf{x}$ is the clean image
- $\tilde{\mathbf{x}}$ is the noisy image ($\tilde{\mathbf{x}} = \mathbf{x} + \sigma \epsilon$, where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$)
- $\sigma$ is the noise level

### 8.2 Mathematical Equivalence

This formula is mathematically equivalent to:

$$
\hat{\omega}_{\text{target}} = \frac{\lVert\epsilon\rVert_2^2}{\sigma}
$$

where $\epsilon = \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma}$

**Proof**:
$$
\frac{\lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2}{\sigma^3} = \frac{\lVert\tilde{\mathbf{x}} - \mathbf{x}\rVert_2^2}{\sigma^3} = \frac{\lVert\sigma \epsilon\rVert_2^2}{\sigma^3} = \frac{\sigma^2 \lVert\epsilon\rVert_2^2}{\sigma^3} = \frac{\lVert\epsilon\rVert_2^2}{\sigma}
$$

### 8.3 Output Transformations

Models trained with different loss functions may output different quantities during training. During evaluation, these outputs are transformed to $\hat{\omega}$ space:

- **Loss Types 1-2** (omega_hat, omega_epsilon): Model outputs $\hat{\omega}$ directly, no transformation needed
- **Loss Type 9** (omega_chi_zscore): Model outputs z-score, transform: $\hat{\omega} = (\text{output} \cdot \sqrt{2d} + d) / \sigma$

where $d$ is the dimensionality of the input (e.g., $d = 3 \times 32 \times 32 = 3072$ for CIFAR-10).

### 8.4 Evaluation Visualizations

The evaluation process generates scatter plots showing:
- **X-axis**: Ground truth $\hat{\omega}_{\text{target}} = \lVert\mathbf{x} - \tilde{\mathbf{x}}\rVert_2^2 / \sigma^3$
- **Y-axis**: Model prediction $\hat{\omega}$ (after applying appropriate output transformation)

These plots are generated for:
1. **Training data** (`train_scatter_predictions.png`) - Generated at the end of training
2. **Validation data** (`val_scatter_predictions.png`) - Generated at the end of training
3. **Test data** (`test_scatter_predictions.png`) - Generated during evaluation

Perfect predictions lie on the diagonal line $y = x$.

---

## Appendix: Practical Usage in Sampling

The trained model $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ is used in the sampling algorithm as follows:

**Algorithm: Sampling with Denoiser and Sigma Score**

1. Initialize: $\sigma_0 = \sigma_{\max}$, $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I})$, $t = 0$

2. While $\sigma_t > \sigma_{\min}$ and $t < T$:
   - Compute sigma score: $\omega_t = \hat{\omega}_\theta(\mathbf{x}_t, \sigma_t) - \frac{d}{\sigma_t}$
   - Update noise level: $\sigma_{t+1} = \sigma_t - \eta_t \cdot \omega_t$
   - Compute data score: $s_\theta(\mathbf{x}_t, \sigma_t) = \frac{D_\theta(\mathbf{x}_t, \sigma_t) - \mathbf{x}_t}{\sigma_t^2}$
   - Update sample: $\mathbf{x}_{t+1} = \mathbf{x}_t + \gamma_t \cdot s_\theta(\mathbf{x}_t, \sigma_t)$
   - $t \leftarrow t + 1$

3. Return $\mathbf{x}_t$

where:
- $D_\theta(\mathbf{x}, \sigma)$ is a pretrained denoising model
- $\eta_t$ and $\gamma_t$ are step sizes
- $\sigma_{\min}$ and $\sigma_{\max}$ define the noise schedule range

This algorithm enables **joint optimization** in $(\mathbf{x}, \sigma)$ space, adaptively controlling the noise level based on the learned sigma score gradient.

---

For implementation details and usage examples, see [USAGE_GUIDE.md](USAGE_GUIDE.md).
