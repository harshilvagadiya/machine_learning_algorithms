# 📐 Discriminant Analysis (LDA & QDA) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Discriminant Analysis is a family of **generative linear and quadratic classifiers** based on Bayes' Theorem. Unlike Logistic Regression (a discriminative model optimizing $P(y \mid \mathbf{x})$ directly), Discriminant Analysis models the class-conditional feature distributions $P(\mathbf{x} \mid y = k)$ as multivariate Gaussians:
$$P(\mathbf{x} \mid y = k) = \frac{1}{(2\pi)^{p/2} |\boldsymbol{\Sigma}_k|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right)$$

Applying Bayes' Rule yields the posterior log-odds:
$$\ln P(y = k \mid \mathbf{x}) = \ln P(\mathbf{x} \mid y = k) + \ln \pi_k - \ln P(\mathbf{x})$$

---

## 2. Linear Discriminant Analysis (LDA) vs Quadratic Discriminant Analysis (QDA)

| Theoretical Dimension | Linear Discriminant Analysis (LDA) | Quadratic Discriminant Analysis (QDA) |
| :--- | :--- | :--- |
| **Covariance Assumption** | Homoscedastic: $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ (Shared covariance) | Heteroscedastic: $\boldsymbol{\Sigma}_k \ne \boldsymbol{\Sigma}_j$ (Class-specific covariance) |
| **Decision Boundary** | Strictly Linear Hyperplane | Curved Quadratic Hypersurface (Ellipsoids / Hyperbolas) |
| **Parameter Complexity** | $\mathcal{O}(p \cdot K + p^2)$ | $\mathcal{O}(K \cdot p^2)$ (Susceptible to curse of dimensionality) |
| **Discriminant Function** | $\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_k + \ln \pi_k$ | $\delta_k(\mathbf{x}) = -\frac{1}{2}\ln|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k) + \ln \pi_k$ |
| **Closed-Form Solution** | Exact MLE (No iterations, zero learning rate) | Exact MLE (Requires non-singular class covariance matrices) |
| **Dimensionality Reduction** | Supervised projection to $K-1$ dimensions | Not directly applicable |

---

## 3. Mathematical Derivations

### Fisher's Linear Discriminant Criterion:
Maximizes the ratio of between-class variance $\mathbf{S}_B$ to within-class variance $\mathbf{S}_W$:
$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}} = \mathbf{S}_W^{-1} (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$$
where:
$$\mathbf{S}_W = \sum_{k=1}^K \sum_{i \in C_k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T, \quad \mathbf{S}_B = \sum_{k=1}^K N_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

---

## 4. Production Engineering & Serving Latency
- **Exact Closed-Form Solution:** Unlike neural networks or logistic regression that require iterative SGD/L-BFGS passes, LDA/QDA solve in $\mathcal{O}(N \cdot p^2 + p^3)$ time via Cholesky decomposition.
- **Inference Latency:** Precomputing $\mathbf{w}_k = \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k$ reduces LDA inference to a single matrix-vector dot product:
  $$\text{Latency} \le 0.15 \text{ ms (P99 < 0.35 ms)}$$
- **Regularization (Ledoit-Wolf / Shrinkage):** When $p > N$, $\boldsymbol{\Sigma}$ is singular. Empirical shrinkage regularizes the matrix:
  $$\boldsymbol{\Sigma}_{\text{shrunk}} = (1 - \lambda)\boldsymbol{\Sigma} + \lambda \frac{\operatorname{Tr}(\boldsymbol{\Sigma})}{p} \mathbf{I}$$

---

## 5. Decision Matrix: When to Deploy
- **Deploy LDA when:** Features are approximately Gaussian, sample size $N$ is small relative to $p$, and strict linear explainability is required.
- **Deploy QDA when:** Classes display drastically different variance structures (e.g. one class is tight and localized while another is broad and dispersed), and $N_k \gg p^2$.
