# 🟡 Distance-Based Family: Master Architecture

Welcome to the **Distance-Based Machine Learning Family**.

Unlike parametric models (Linear Regression, Logistic Regression, SGD) that fit a global linear equation ($\\mathbf{w}^T \\mathbf{x} + b$), **Distance-Based Models** make predictions purely based on the spatial proximity of data points in high-dimensional geometric space.

---

## 🧭 Algorithms in the Distance-Based Family

```
                           DISTANCE-BASED FAMILY
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
DISCRETE TARGET (CLASSIFICATION)                    CONTINUOUS TARGET (REGRESSION)
"Konsi class mein belong karta hai?"                "Continuous numerical value kya hogi?"
           │                                                   │
           ▼                                                   ▼
01_KNN_Classifier                                   02_KNN_Regressor
(KNeighborsClassifier)                              (KNeighborsRegressor)
- Majority / Distance-Weighted Voting               - Local Mean / Distance-Weighted Average
- Soft Probs via Neighbor Fractions                 - Continuous Local Interpolation
- Non-linear decision boundaries                    - Non-linear regression curves
```

---

## 📑 Family Directory Navigation

1. 📂 **[`01_KNN_Classifier/`](01_KNN_Classifier/README.md)** — Complete 7-guide classification suite.
2. 📂 **[`02_KNN_Regressor/`](02_KNN_Regressor/README.md)** — Complete 7-guide regression suite.

---

## ⚡ The Golden Laws of Distance-Based Learning

1. **Feature Scaling is 100% Mandatory:** Unscaled high-magnitude features dominate the Euclidean $L_2$ norm, blinding the model to lower-magnitude signals.
2. **Curse of Dimensionality:** When features $D > 20$, points become equidistant ($\\lim_{D\\to\\infty} \\frac{d_{\\max}-d_{\\min}}{d_{\\min}} \\to 0$). Always apply feature selection or PCA.
3. **No Extrapolation:** Distance-based models cannot extrapolate outside the bounding box of their training data!
