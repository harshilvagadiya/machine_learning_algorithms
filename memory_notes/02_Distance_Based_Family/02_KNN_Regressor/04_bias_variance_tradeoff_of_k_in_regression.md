# Guide 04: The Bias-Variance Tradeoff of $K$ in Regression

> **Note on Workflow:**
> For standard cross-validation, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details **how hyperparameter $K$ controls overfitting vs underfitting in regression**.

---

## 1. The Role of $K$ in Regression

In linear regression, model complexity is controlled by regularizers ($\\alpha, \\lambda$).
In KNN Regression, **$K$ is the primary regularization knob**:

```
Small K (K=1) ──────────────────────────────────────────────> Large K (K=N)
Ultra-Complex, Jagged Function                              Completely Flat Constant Line
High Variance / Zero Bias                                   Low Variance / High Bias
Overfitting (Memorizes Noise)                               Underfitting (Global Mean)
```

---

## 2. Detailed Breakdown Across $K$

| Parameter $K$ | Regression Surface | Train $R^2$ / MSE | Test $R^2$ / MSE | Risk |
| :--- | :--- | :--- | :--- | :--- |
| **$K = 1$** | Curve passes through **every single training point**. Perfect memorization. | $R^2 = 1.0$ (MSE $= 0.0$) | Very Poor (High MSE) | **Severe Overfitting** (Fits random noise/outliers) |
| **$K = \\sqrt{N}$** | Smooth local averaging. Individual noise spikes are canceled out. | Moderate | **Optimal** (Lowest Test MSE) | **Balanced Generalization** |
| **$K \\to N$** | Predicts the exact **Global Sample Mean ($\\bar{y}$)** for every query point! | $R^2 \\to 0.0$ | $R^2 \\to 0.0$ | **Severe Underfitting** |

```
    K = 1 (Overfitting / Noise Fitting)          K = 15 (Smooth / Generalized)
    y ^       .                                  y ^
      |      / \      .                           |         . - - .
      |  .  /   \    / \                         |       .         .
      | / \/     \  /   \                        |   . -             - .
      +-------------------------> x              +-------------------------> x
      Passes through every outlier!              Smooth regression trendline.
```

---

## 3. Finding Optimal $K$ via Validation Curves

When tuning $K$, always plot **Validation Mean Squared Error (MSE)** vs $K$:
* As $K$ increases from 1 to 5-15, Validation Error drops sharply (variance reduces).
* Past the optimal $K$, Validation Error starts climbing again (bias increases).
* The bottom of the U-curve gives the best $K$!
