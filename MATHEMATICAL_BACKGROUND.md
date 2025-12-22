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
8. [References](#references)

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

### Loss Type 3: `omega_chi_approx` (Chi-Squared Approximation)

**Description**: Uses the normal approximation of the chi-squared distribution.

**Target**:
$$
\hat{\omega}_{\text{target}} = \frac{d + \sqrt{2d} \cdot Z}{\sigma}, \quad Z \sim \mathcal{N}(0, 1)
$$

**Loss Function**:
$$
\mathcal{L}_3 = \left(\hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{d + \sqrt{2d} \cdot Z}{\sigma}\right)^2
$$

**Note**: $Z$ is sampled from standard normal distribution during training.

---

### Loss Type 4: `omega_chi_mean` (Expected Chi-Squared)

**Description**: Uses the expected value of $\lVert\epsilon\rVert_2^2$.

**Target**:
$$
\hat{\omega}_{\text{target}} = \frac{d}{\sigma}
$$

**Loss Function**:
$$
\mathcal{L}_4 = \left(\hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{d}{\sigma}\right)^2
$$

**Advantage**: Deterministic target, no stochastic sampling required.

---

### Loss Type 5: `sigma_direct` (Direct Sigma Estimation)

**Description**: Introduces $\acute{\omega}_\theta$ which directly estimates $\sigma$.

**Definition**:
$$
\hat{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\acute{\omega}_\theta(\mathbf{x}, \sigma)}
$$

**Transformation**:
$$
\acute{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\hat{\omega}_\theta(\mathbf{x}, \sigma)}
$$

**Loss Function**:
$$
\mathcal{L}_5 = \left(\acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma\right)^2
$$

**Interpretation**: The network learns to predict $\sigma$ directly, which is then inverted to obtain $\hat{\omega}$.

---

### Loss Type 6: `sigma_normalized` (Normalized Sigma Estimation)

**Description**: Normalized version of direct sigma estimation.

**Loss Function**:
$$
\mathcal{L}_6 = \frac{\left(\acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma\right)^2}{\sigma}
$$

**Advantage**: Normalizes the loss by $\sigma$, giving more weight to errors at small noise levels.

---

### Loss Type 7: `sigma_relative` (Relative Sigma Estimation)

**Description**: Relative error formulation for sigma estimation.

**Loss Function**:
$$
\mathcal{L}_7 = \frac{\left(\acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma\right)^2}{\sigma^2}
$$

**Advantage**: Emphasizes relative error, making the loss scale-invariant with respect to $\sigma$.

---

### Loss Type 8: `sigma_calibrated` (Calibrated Sigma Estimation)

**Description**: Calibrated formulation similar to Sigma-Cal Loss [4].

**Loss Function**:
$$
\mathcal{L}_8 = \left(\acute{\omega}_\theta(\mathbf{x}, \sigma) - (\sigma_{\text{cal}} - \sigma)\right)^2
$$

**Note**: $\sigma_{\text{cal}}$ is a calibration parameter that can be learned or set empirically.

---

### Summary Table

| Loss Type | Name | Target Quantity | Key Feature | Estimates |
|-----------|------|----------------|-------------|-----------|
| 1 | `omega_hat` | $\frac{\\|\mathbf{x} - \tilde{\mathbf{x}}\\|^2}{\sigma^3}$ | Corrected formulation | $\nabla_\sigma \log[\sigma^d p]$ |
| 2 | `omega_epsilon` | $\frac{\\|\epsilon\\|^2}{\sigma}$ | Noise-based | $\nabla_\sigma \log[\sigma^d p]$ |
| 3 | `omega_chi_approx` | $\frac{d + \sqrt{2d} Z}{\sigma}$ | Chi-squared approximation | $\nabla_\sigma \log[\sigma^d p]$ |
| 4 | `omega_chi_mean` | $\frac{d}{\sigma}$ | Expected value | $\nabla_\sigma \log[\sigma^d p]$ |
| 5 | `sigma_direct` | $\sigma$ | Direct sigma prediction | $\nabla_\sigma \log[\sigma^d p]$ |
| 6 | `sigma_normalized` | $\sigma$ (normalized) | Weighted by $1/\sigma$ | $\nabla_\sigma \log[\sigma^d p]$ |
| 7 | `sigma_relative` | $\sigma$ (relative) | Scale-invariant | $\nabla_\sigma \log[\sigma^d p]$ |
| 8 | `sigma_calibrated` | $\sigma_{\text{cal}} - \sigma$ | Calibrated prediction | $\nabla_\sigma \log[\sigma^d p]$ |

---

## 7. Implementation Notes

### 7.1 Model Output Interpretation

For **Loss Types 1-4**, the model directly outputs $\hat{\omega}_\theta(\mathbf{x}, \sigma)$ which estimates $\nabla_\sigma \log[\sigma^d p(\mathbf{x}, \sigma)]$.

For **Loss Types 5-8**, the model output is transformed:
$$
\hat{\omega}_\theta \rightarrow \acute{\omega}_\theta = \frac{d}{\hat{\omega}_\theta} \rightarrow \hat{\omega}_\theta = \frac{d}{\acute{\omega}_\theta} = \hat{\omega}_\theta
$$

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

- For Loss Types 5-8, ensure $\hat{\omega}_\theta > \epsilon$ (small positive constant) to avoid division by zero
- Use appropriate activation functions (e.g., softplus, exponential with offset) to ensure positive outputs
- For very small $\sigma$ values, consider using log-space computations

### 7.4 Training Considerations

- **Loss Types 1-4**: Direct estimation approaches, suitable for most applications
- **Loss Type 4**: Simplest formulation with deterministic target, good baseline
- **Loss Types 5-8**: Indirect estimation via $\sigma$ prediction, may provide better numerical properties
- The corrected objective (using $\hat{\omega}_\theta$) ensures monotonic decrease of $\sigma$ during sampling

---

## 8. References

[1] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-Based Generative Modeling through Stochastic Differential Equations. *International Conference on Learning Representations (ICLR)*.

[2] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 26565-26577.

[3] Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.

[4] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 6840-6851.

[5] Vincent, P. (2011). A Connection Between Score Matching and Denoising Autoencoders. *Neural Computation*, 23(7), 1661-1674.

[6] Hyvärinen, A., & Dayan, P. (2005). Estimation of Non-Normalized Statistical Models by Score Matching. *Journal of Machine Learning Research*, 6, 695-709.

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
