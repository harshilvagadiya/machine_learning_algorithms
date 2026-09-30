# 🎲 Bayesian Regression (Bayesian Ridge & ARD) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Standard Ridge and Lasso regression provide single point estimates $\hat{\mathbf{w}}$ minimizing a penalized sum of squares.
**Bayesian Linear Regression** formulates regression within a probabilistic framework, treating the weights $\mathbf{w}$ not as fixed deterministic parameters, but as **random variables governed by prior probability distributions**.

The model outputs a full **posterior predictive distribution**:
$$P(y_* \mid \mathbf{x}_*, \mathbf{X}, \mathbf{y}) = \mathcal{N}\left(\mu_*(\mathbf{x}_*), \sigma_*^2(\mathbf{x}_*)\right)$$
delivering principled, closed-form predictive uncertainties for every single inference query.

---

## 2. Bayesian Ridge vs Automatic Relevance Determination (ARD)

| Dimension | Bayesian Ridge Regression | Automatic Relevance Determination (ARD) |
| :--- | :--- | :--- |
| **Prior on Weights** | Spherical Gaussian: $P(\mathbf{w} \mid \alpha) = \mathcal{N}(\mathbf{0}, \alpha^{-1} \mathbf{I})$ | Component-wise Gaussian: $P(\mathbf{w} \mid \boldsymbol{\alpha}) = \mathcal{N}(\mathbf{0}, \operatorname{diag}(\alpha_1^{-1}, \dots, \alpha_p^{-1}))$ |
| **Hyperparameters** | Single scalar precision $\alpha$, noise precision $\lambda$ | $p$ distinct feature precisions $\alpha_j$, noise precision $\lambda$ |
| **Feature Sparsity** | Smooth shrinkage (Bayesian $L_2$ equivalent) | **Sparsity-inducing:** As $\alpha_j \to \infty$, $w_j \to 0$ (Bayesian Lasso equivalent) |
| **Optimization Method** | MacKay Evidence Approximation / Empirical Bayes | MacKay / Expectation-Maximization on Type-II Marginal Likelihood |
| **Tuning Requirement** | Zero hyperparameter cross-validation needed | Automatic feature selection via empirical marginal likelihood |

---

## 3. Mathematical Foundations

### Prior and Likelihood:
$$P(\mathbf{y} \mid \mathbf{X}, \mathbf{w}, \lambda) = \mathcal{N}(\mathbf{X}\mathbf{w}, \lambda^{-1}\mathbf{I})$$
$$P(\mathbf{w} \mid \boldsymbol{\alpha}) = \mathcal{N}(\mathbf{0}, \mathbf{A}^{-1}), \quad \mathbf{A} = \operatorname{diag}(\alpha_1, \dots, \alpha_p)$$

### Exact Posterior Distribution:
By Gaussian conjugacy, the posterior over weights is analytically Gaussian:
$$P(\mathbf{w} \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol{\mu}_N, \boldsymbol{\Sigma}_N)$$
$$\boldsymbol{\Sigma}_N = \left( \mathbf{A} + \lambda \mathbf{X}^T \mathbf{X} \right)^{-1}$$
$$\boldsymbol{\mu}_N = \lambda \boldsymbol{\Sigma}_N \mathbf{X}^T \mathbf{y}$$

### Predictive Distribution at Query $\mathbf{x}_*$:
$$\mathbb{E}[y_* \mid \mathbf{x}_*] = \mathbf{x}_*^T \boldsymbol{\mu}_N$$
$$\operatorname{Var}(y_* \mid \mathbf{x}_*) = \lambda^{-1} + \mathbf{x}_*^T \boldsymbol{\Sigma}_N \mathbf{x}_*$$

---

## 4. Production Engineering & Serving Latency
- **Self-Tuning Regularization:** $\alpha$ and $\lambda$ are estimated directly from training data by maximizing the log marginal likelihood (Type-II Maximum Likelihood). Grid search cross-validation is unnecessary.
- **Inference SLA:** Serving uses pre-computed mean $\boldsymbol{\mu}_N$ ($\mathcal{O}(p)$ time) and optional predictive variance $\mathbf{x}_*^T \boldsymbol{\Sigma}_N \mathbf{x}_*$ ($\mathcal{O}(p^2)$ time):
  $$\text{Mean Inference Latency} \le 0.25 \text{ ms (P99 < 0.60 ms)}$$
- **Safety Critical Systems:** High predictive variance $\sigma_*^2$ flags out-of-distribution inputs in autonomous and financial pipelines.
