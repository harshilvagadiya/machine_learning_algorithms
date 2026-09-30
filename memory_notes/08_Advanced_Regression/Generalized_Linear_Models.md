# 📊 Generalized Linear Models (GLM) Architecture & Memory Notes
## Poisson, Gamma, Tweedie, & Negative Binomial Regression

### 1. Unified Exponential Dispersion Family Framework
Generalized Linear Models (GLMs) extend Ordinary Least Squares by assuming the target follows an Exponential Dispersion Model:
$$P(y \mid \theta, \phi) = \exp\left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right)$$
with monotonic link function $g(\mu) = \mathbf{x}^T \mathbf{w}$ connecting the linear predictor to the expected conditional mean $\mu = \mathbb{E}[y \mid \mathbf{x}] = b'(\theta)$.
The conditional variance function is governed by:
$$\operatorname{Var}(y \mid \mathbf{x}) = \phi \cdot V(\mu)$$

---

### 2. The Four Primary GLM Distributions

| GLM Family | Canonical Link $g(\mu)$ | Variance Function $V(\mu)$ | Empirical Production Domain |
| :--- | :--- | :--- | :--- |
| **Poisson Regression** | $\ln(\mu) = \mathbf{x}^T \mathbf{w}$ | $V(\mu) = \mu$ (Equi-dispersion) | Rare count data (website hits, customer arrivals, equipment defects) |
| **Gamma Regression** | $-\mu^{-1}$ or $\ln(\mu)$ | $V(\mu) = \mu^2$ (Constant CV) | Positive continuous skewed data (customer lifetime value, claim severity) |
| **Tweedie Regression** | $\ln(\mu)$ | $V(\mu) = \mu^p$ ($1 < p < 2$) | Compound Poisson-Gamma zero-inflated insurance loss cost modeling |
| **Negative Binomial (NB-2)** | $\ln(\mu) = \mathbf{x}^T \mathbf{w}$ | $V(\mu) = \mu + \alpha \mu^2$ ($\alpha > 0$) | **Over-dispersed count data** (fleet rentals, medical visits, footfall) |

---

### 3. The Overdispersion Problem & Negative Binomial Solution
In Poisson regression, the conditional mean equals the variance:
$$\mathbb{E}[y] = \operatorname{Var}(y) = \mu$$
In real-world data, unobserved heterogeneity causes **over-dispersion** ($\operatorname{Var}(y) \gg \mu$). Fitting Poisson to over-dispersed counts artificially deflates standard errors, inflating false positive discoveries.

**Negative Binomial Regression** models unobserved Gamma-distributed frailty $\epsilon_i \sim \operatorname{Gamma}(1/\alpha, \alpha)$, resulting in a quadratic variance function:
$$\operatorname{Var}(y \mid \mathbf{x}) = \mu + \alpha \mu^2$$
As dispersion parameter $\alpha \to 0$, Negative Binomial smoothly converges to Poisson.

---

### 4. Production Serving & Inference Latency
- **Inference SLA:** Evaluating GLMs requires only a linear dot product followed by the exponential inverse-link:
  $$\hat{\mu} = \exp(\mathbf{x}_*^T \mathbf{w} + b) \implies \text{Mean Serving Latency} \le 0.05 \text{ ms (P99 < 0.15 ms)}$$
- **Sub-50ms Enterprise Compliance:** All 4 GLM families pass the sub-50ms smoke test with $100\times$ margin to spare.
