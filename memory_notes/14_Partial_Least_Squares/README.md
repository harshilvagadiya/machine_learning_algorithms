# 🧪 Partial Least Squares (PLS Regression & PLS Canonical) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Principal Component Regression (PCR) projects features $\mathbf{X}$ onto directions of maximum feature variance (unsupervised PCA) and then fits OLS regression. However, directions of maximal variance in $\mathbf{X}$ may have zero correlation with target $\mathbf{Y}$.

**Partial Least Squares (PLS)** (Wold, 1975) is a **supervised bilinear dimensionality reduction algorithm**. It finds orthogonal latent score vectors $\mathbf{t} = \mathbf{X}\mathbf{w}$ and $\mathbf{u} = \mathbf{Y}\mathbf{q}$ that **maximize the sample covariance between $\mathbf{X}$ and $\mathbf{Y}$ simultaneously**:
$$\max_{\mathbf{w}, \mathbf{q}} \operatorname{Cov}^2(\mathbf{X}\mathbf{w}, \mathbf{Y}\mathbf{q}) \quad \text{subject to } \|\mathbf{w}\|_2 = 1, \|\mathbf{q}\|_2 = 1$$

---

## 2. PLS Regression (PLS1 / PLS2) vs PLS Canonical (CCA)

| Architectural Dimension | PLSRegression (PLS-R) | PLSCanonical (PLS-C) |
| :--- | :--- | :--- |
| **Objective** | Asymmetric prediction: Predict $\mathbf{Y}$ from $\mathbf{X}$ | Symmetric association: Maximize bidirectional correlation between $\mathbf{X}$ and $\mathbf{Y}$ |
| **Deflation Protocol** | Asymmetric deflation of $\mathbf{X}$ using score $\mathbf{t}$ | Symmetric deflation of both $\mathbf{X}$ and $\mathbf{Y}$ using their respective scores |
| **Component Constraint** | $n\_components \le \min(N, p)$ | $n\_components \le \min(N, p, q)$ (Bounded by number of targets $q$) |
| **Use Cases** | Chemometrics, spectroscopy, econometrics with collinear features | Multi-omics genomics, neuroscience (cross-modal brain-behavior correlations) |

---

## 3. Mathematical Foundations: The NIPALS Algorithm
Non-linear Iterative Partial Least Squares (NIPALS) solves for components iteratively:
1. Initialize $\mathbf{u}$ with a column of $\mathbf{Y}$.
2. Compute $\mathbf{X}$-weights: $\mathbf{w} = \frac{\mathbf{X}^T \mathbf{u}}{\|\mathbf{X}^T \mathbf{u}\|_2}$.
3. Compute $\mathbf{X}$-scores: $\mathbf{t} = \mathbf{X}\mathbf{w}$.
4. Compute $\mathbf{Y}$-weights: $\mathbf{q} = \frac{\mathbf{Y}^T \mathbf{t}}{\|\mathbf{Y}^T \mathbf{t}\|_2}$.
5. Update $\mathbf{u} = \mathbf{Y}\mathbf{q}$. Iterate until $\|\mathbf{t}_{\text{new}} - \mathbf{t}_{\text{old}}\| < \epsilon$.
6. Deflate matrices:
   $$\mathbf{X} \leftarrow \mathbf{X} - \mathbf{t}\mathbf{p}^T, \quad \mathbf{p} = \frac{\mathbf{X}^T \mathbf{t}}{\mathbf{t}^T \mathbf{t}}$$
   $$\mathbf{Y} \leftarrow \mathbf{Y} - \mathbf{t}\mathbf{c}^T, \quad \mathbf{c} = \frac{\mathbf{Y}^T \mathbf{t}}{\mathbf{t}^T \mathbf{t}}$$

The final regression matrix is given by:
$$\mathbf{B}_{\text{PLS}} = \mathbf{W}(\mathbf{P}^T \mathbf{W})^{-1}\mathbf{Q}^T$$

---

## 4. Production Engineering & Serving Latency
- **Solving Massive Multicollinearity ($p \gg N$):** When features are collinear (e.g. 2,000 spectral wavelengths on 150 blood samples), OLS fails due to singular $(\mathbf{X}^T\mathbf{X})^{-1}$. PLS compresses collinear features into 2 to 10 orthogonal latent scores without matrix inversion failure.
- **Inference Latency:** Precomputing $\mathbf{B}_{\text{PLS}}$ condenses inference to a single linear matrix multiplication:
  $$\hat{\mathbf{y}} = \mathbf{x}^T \mathbf{B}_{\text{PLS}} + \mathbf{b}_0 \implies \text{Latency} \le 0.10 \text{ ms (P99 < 0.25 ms)}$$
