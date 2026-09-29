# Guide 01: The Local Mean & Non-Parametric Regression Intuition

> **Note on Workflow:**
> For universal pre-flight steps (Data Ingestion, Missing Values, Train/Test Split, ColumnTransformer), refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **core mathematical intuition of K-Nearest Neighbors Regression**.

---

## 1. How KNN Regression Works

In Linear Regression, the model fits a straight line or global hyperplane through the entire dataset:
$$\\hat{y} = w_1 x_1 + w_2 x_2 + \\dots + b$$

In **K-Nearest Neighbors Regression (`KNeighborsRegressor`)**, there is **no line, no slope ($w$), and no intercept ($b$)**!
Instead of a global formula, KNN makes a **purely local estimate**:

```
[New Query Point x*]
        │
        ▼
Calculate distance d(x*, x_i) to all training samples
        │
        ▼
Identify the K closest neighbors: N_K(x*)
        │
        ▼
Compute Local Mean of their target values:
y_hat = (1 / K) * sum(y_i for i in N_K(x*))
```

---

## 2. Real-World Analogy: Real Estate Appraisals

Imagine you want to estimate the price of a house:
* **Linear Regression Approach:** Uses a fixed national formula:
  $$\\text{Price} = \\$150 \\times \\text{SqFt} + \\$20,000 \\times \\text{Bedrooms}$$
  *(Fails if houses in different neighborhoods don't follow this exact line).*
* **KNN Regression Approach (Real Estate Appraiser):**
  An appraiser does not use a rigid formula. They look at the **3 to 5 most comparable houses recently sold in the exact same neighborhood**:
  $$\\hat{y} = \\frac{\\$450,000 + \\$460,000 + \\$455,000}{3} = \\$455,000$$

---

## 3. Parametric vs Non-Parametric Regression

| Property | Linear / Ridge / Lasso Regression | K-Nearest Neighbors Regression |
| :--- | :--- | :--- |
| **Model Nature** | Parametric (Fixed weights $\\mathbf{w} \\in \\mathbb{R}^D$) | Non-Parametric (Grows with training data) |
| **Training Time** | $O(N \\cdot D)$ | $O(1)$ (Instantaneous storage) |
| **Inference Time** | $O(D)$ (Instant dot product) | $O(N \\cdot D)$ (Must calculate distance to all training rows) |
| **Function Shape** | Strictly linear / planar | Can fit **any non-linear, sinusoidal, curved, or disjoint function**! |
| **Model Size on Disk** | 5 KB - 50 KB | Equal to the size of the training dataset |
