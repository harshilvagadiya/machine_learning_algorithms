# Guide 05: Regression Evaluation Metrics & Residual Diagnostics

> **Note on Workflow:**
> For general evaluation rules, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **evaluation metrics and residual diagnostics specific to KNN Regression**.

---

## 1. Core Evaluation Metrics for KNN Regression

$$R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2}$$

$$\\text{MAE} = \\frac{1}{N} \\sum |y_i - \\hat{y}_i| \\qquad \\text{RMSE} = \\sqrt{\\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2}$$

* **$R^2$ Score:** Percentage of variance explained by the local neighborhoods.
* **MAE (Mean Absolute Error):** Average dollar/unit error in real human terms.
* **RMSE (Root Mean Squared Error):** Heavily penalizes large prediction mistakes.

---

## 2. Residual Diagnostics for KNN

A residual is the error for sample $i$:
$$e_i = y_i - \\hat{y}_i$$

### What to Look For in KNN Residual Plots:
1. **Homoscedasticity (Equal Variance):** Residuals should bounce randomly around zero with constant spread.
2. **Boundary Residual Spikes (The Edge Effect):**
   * Look closely at the edges (lowest $y$ and highest $y$).
   * Because KNN cannot extrapolate, residuals at the maximum training boundaries will consistently show **positive bias** (underestimating extreme high values)!
