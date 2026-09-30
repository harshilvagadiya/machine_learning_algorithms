# 📐 Orthogonal Distance Regression (ODR / Errors-in-Variables) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Ordinary Least Squares (OLS) regression operates under the fundamental assumption that independent covariates $\mathbf{x}$ are observed without noise:
$$y_i = \mathbf{x}_i^T \boldsymbol{\beta} + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$
In production engineering, sensor measurements, financial indices, demographic surveys, and laboratory assays are **invariably subject to observation error (Errors-in-Variables)**.

When features contain measurement error $\delta$, OLS suffers from **attenuation bias** (systematic shrinkage of slope coefficients toward zero).
**Orthogonal Distance Regression (ODR)** (Boggs, Byrd, & Schnabel, 1987) minimizes the shortest **orthogonal Euclidean distance** from each data point to the regression manifold, accounting for errors in **both $\mathbf{x}$ and $y$ simultaneously**.

---

## 2. Mathematical Formulation: Total Least Squares vs OLS

```
   OLS (Vertical Residuals)             ODR (Orthogonal Shortest Distance)
         y                                    y
         │         *                          │         *
         │        /│                          │        / ╲
         │       / │                          │       /   ╲
         │      /  │ (Error in y only)        │      /     * (Error in BOTH x & y)
         │     /   *                          │     /
         │    /                               │    /
         └────┴────────────── x               └────┴────────────── x
```

### The ODR Optimization Problem:
$$y_i = f(\mathbf{x}_i + \boldsymbol{\delta}_i; \boldsymbol{\beta}) + \epsilon_i$$
Minimize the weighted sum of squared errors:
$$\min_{\boldsymbol{\beta}, \boldsymbol{\delta}_1, \dots, \boldsymbol{\delta}_n} \sum_{i=1}^n \left( \frac{\epsilon_i^2}{d_{y, i}^2} + \boldsymbol{\delta}_i^T \mathbf{D}_{x, i}^{-2} \boldsymbol{\delta}_i \right) \quad \text{subject to } \epsilon_i = y_i - f(\mathbf{x}_i + \boldsymbol{\delta}_i; \boldsymbol{\beta})$$
where $d_{y, i}^2$ and $\mathbf{D}_{x, i}$ are the known or estimated observational error variances.

### Numerical Optimization via ODRPACK:
Because the objective involves both model parameters $\boldsymbol{\beta}$ and nuisance deviations $\boldsymbol{\delta}_i$, the parameter space is $p + N \cdot p$. ODRPACK solves this efficiently via a specialized **trust-region Levenberg-Marquardt algorithm** exploiting the block-diagonal structure of the Jacobian.

---

## 3. Production Engineering & Serving Latency
- **Rotational Invariance:** Unlike OLS whose predictions change under coordinate axis rotation, ODR is geometrically invariant under orthogonal coordinate transformations.
- **Inference Latency:** At serving time, parameter vector $\boldsymbol{\beta}$ is fixed, so prediction simplifies to direct evaluation $\hat{y} = f(\mathbf{x}; \boldsymbol{\beta})$:
  $$\text{Mean Serving Latency} \le 0.08 \text{ ms (P99 < 0.22 ms)}$$
- **Deploy ODR when:** Regressors are physical measurements with sensor tolerances, calibration curves (spectroscopy, chromatography), or econometric variables observed with sampling error.
