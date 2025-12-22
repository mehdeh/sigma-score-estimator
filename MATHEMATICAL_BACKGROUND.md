# Mathematical Background: Omega Estimation

## Overview

This document provides the mathematical foundation for estimating $\omega(\mathbf{x}, \sigma)$, which is derived from the score function of noisy data. We present multiple loss function formulations for training neural networks to estimate $\omega$ or related quantities.

## Core Definitions

### Score Function and Omega

Given clean data $\tilde{\mathbf{x}}$ and noisy observation $\mathbf{x} \sim \mathcal{N}(\tilde{\mathbf{x}}, \sigma^2 \mathbf{I})$, the score function is:

$$
\nabla_{\mathbf{x}} \log p(\mathbf{x} | \sigma) = -\frac{\mathbf{x} - \tilde{\mathbf{x}}}{\sigma^2}
$$

The quantity $\omega$ is defined as:

$$
\omega(\mathbf{x}, \sigma) = \lVert \nabla_{\mathbf{x}} \log p(\mathbf{x} | \sigma) \rVert_2^2 = \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^4}
$$

### Noise Parameterization

Using the reparameterization trick, we can write:

$$
\mathbf{x} = \tilde{\mathbf{x}} + \sigma \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

Therefore:

$$
\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2 = \sigma^2 \lVert \epsilon \rVert_2^2
$$

### Chi-Squared Distribution Properties

Since $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ with dimension $d$ (e.g., $d = 3 \times 32 \times 32 = 3072$ for CIFAR-10), we have:

$$
\lVert \epsilon \rVert_2^2 \sim \chi^2_d
$$

**Key properties:**
- **Expectation**: $\mathbb{E}[\lVert \epsilon \rVert_2^2] = d$
- **Variance**: $\text{Var}[\lVert \epsilon \rVert_2^2] = 2d$
- **Normal approximation**: $\lVert \epsilon \rVert_2^2 \approx d + \sqrt{2d} \cdot Z$, where $Z \sim \mathcal{N}(0, 1)$

---

## Loss Function Formulations

We present eight different loss function formulations for training neural networks to estimate $\omega$ or related quantities.

### Loss Type 1: `omega_hat` (Original Formulation)

**Description**: Direct estimation of $\omega$ using the exact form.

**Target**:
$$
\hat{\omega}_{\text{target}} = \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3}
$$

**Loss Function**:
$$
\mathcal{L}_1 = \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert \mathbf{x} - \tilde{\mathbf{x}} \rVert_2^2}{\sigma^3} \right)^2
$$

**Note**: This is the baseline formulation with no approximations.

---

### Loss Type 2: `omega_epsilon` (Epsilon-Based Formulation)

**Description**: Estimation of $\omega$ using the noise $\epsilon$ directly.

**Target** (using $\mathbf{x} = \tilde{\mathbf{x}} + \sigma \epsilon$):
$$
\hat{\omega}_{\text{target}} = \frac{\lVert \epsilon \rVert_2^2}{\sigma}
$$

**Loss Function**:
$$
\mathcal{L}_2 = \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{\lVert \epsilon \rVert_2^2}{\sigma} \right)^2
$$

**Advantage**: More numerically stable when $\sigma$ is small.

---

### Loss Type 3: `omega_chi_approx` (Chi-Squared Approximation)

**Description**: Uses the normal approximation of the chi-squared distribution.

**Target** (using $\lVert \epsilon \rVert_2^2 \approx d + \sqrt{2d} \cdot Z$):
$$
\hat{\omega}_{\text{target}} = \frac{d + \sqrt{2d} \cdot Z}{\sigma}, \quad Z \sim \mathcal{N}(0, 1)
$$

**Loss Function**:
$$
\mathcal{L}_3 = \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{d + \sqrt{2d} \cdot Z}{\sigma} \right)^2
$$

**Note**: $Z$ is sampled from standard normal distribution during training.

---

### Loss Type 4: `omega_chi_mean` (Expected Chi-Squared)

**Description**: Uses the expected value of $\lVert \epsilon \rVert_2^2$.

**Target** (using $\mathbb{E}[\lVert \epsilon \rVert_2^2] = d$):
$$
\hat{\omega}_{\text{target}} = \frac{d}{\sigma}
$$

**Loss Function**:
$$
\mathcal{L}_4 = \left( \hat{\omega}_\theta(\mathbf{x}, \sigma) - \frac{d}{\sigma} \right)^2
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
\acute{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\text{output}_\theta(\mathbf{x}, \sigma)}
$$

**Loss Function**:
$$
\mathcal{L}_5 = \left( \acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma \right)^2
$$

**Interpretation**: The network learns to predict $\sigma$ directly, which is then inverted to obtain $\omega$.

---

### Loss Type 6: `sigma_normalized` (Normalized Sigma Estimation)

**Description**: Normalized version of direct sigma estimation.

**Transformation**:
$$
\acute{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\text{output}_\theta(\mathbf{x}, \sigma)}
$$

**Loss Function**:
$$
\mathcal{L}_6 = \frac{\left( \acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma \right)^2}{\sigma}
$$

**Advantage**: Normalizes the loss by $\sigma$, giving more weight to errors at small noise levels.

---

### Loss Type 7: `sigma_relative` (Relative Sigma Estimation)

**Description**: Relative error formulation for sigma estimation.

**Transformation**:
$$
\acute{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\text{output}_\theta(\mathbf{x}, \sigma)}
$$

**Loss Function**:
$$
\mathcal{L}_7 = \frac{\left( \acute{\omega}_\theta(\mathbf{x}, \sigma) - \sigma \right)^2}{\sigma^2}
$$

**Advantage**: Emphasizes relative error, making the loss scale-invariant with respect to $\sigma$.

---

### Loss Type 8: `sigma_calibrated` (Calibrated Sigma Estimation)

**Description**: Calibrated formulation similar to Sigma-Cal Loss.

**Transformation**:
$$
\acute{\omega}_\theta(\mathbf{x}, \sigma) = \frac{d}{\text{output}_\theta(\mathbf{x}, \sigma)}
$$

**Loss Function**:
$$
\mathcal{L}_8 = \left( \acute{\omega}_\theta(\mathbf{x}, \sigma) - (\sigma_{\text{cal}} - \sigma) \right)^2
$$

**Note**: $\sigma_{\text{cal}}$ is a calibration parameter that can be learned or set empirically. In the simplest case, $\sigma_{\text{cal}} = 0$ reduces to a sign-inverted version of Loss Type 5.

---

## Summary Table

| Loss Type | Name | Target Quantity | Key Feature |
|-----------|------|----------------|-------------|
| 1 | `omega_hat` | $\frac{\\|\mathbf{x} - \tilde{\mathbf{x}}\\|^2}{\sigma^3}$ | Original formulation |
| 2 | `omega_epsilon` | $\frac{\\|\epsilon\\|^2}{\sigma}$ | Noise-based |
| 3 | `omega_chi_approx` | $\frac{d + \sqrt{2d} Z}{\sigma}$ | Chi-squared approximation |
| 4 | `omega_chi_mean` | $\frac{d}{\sigma}$ | Expected value |
| 5 | `sigma_direct` | $\sigma$ | Direct sigma prediction |
| 6 | `sigma_normalized` | $\sigma$ (normalized) | Weighted by $1/\sigma$ |
| 7 | `sigma_relative` | $\sigma$ (relative) | Scale-invariant |
| 8 | `sigma_calibrated` | $\sigma_{\text{cal}} - \sigma$ | Calibrated prediction |

---

## Implementation Notes

### Model Output Interpretation

For **Loss Types 1-4**, the model directly outputs $\hat{\omega}_\theta(\mathbf{x}, \sigma)$.

For **Loss Types 5-8**, the model output is transformed:
$$
\text{output}_\theta \rightarrow \acute{\omega}_\theta = \frac{d}{\text{output}_\theta}
$$

where $d$ is the image dimensionality (e.g., $d = 3072$ for CIFAR-10 images of size $32 \times 32 \times 3$).

### Numerical Stability

- For Loss Types 5-8, ensure $\text{output}_\theta > \epsilon$ (small positive constant) to avoid division by zero.
- Use appropriate activation functions (e.g., softplus, exponential) to ensure positive outputs.

### Training Considerations

- Loss Types 1-2 estimate $\omega$ directly and require $\sigma^3$ or $\sigma$ normalization.
- Loss Type 3 introduces stochasticity through sampling $Z \sim \mathcal{N}(0, 1)$.
- Loss Type 4 provides a deterministic target based on expectation.
- Loss Types 5-8 estimate $\sigma$ (or related quantities) and are then inverted to obtain $\omega$.

---

## References

This formulation builds upon score matching and denoising techniques in diffusion models, extending the framework to explicitly estimate the magnitude of the score function.
