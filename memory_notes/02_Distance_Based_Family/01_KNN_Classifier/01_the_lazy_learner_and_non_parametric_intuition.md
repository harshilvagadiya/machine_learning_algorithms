# Guide 01: The "Lazy Learner" & Non-Parametric Intuition

> **Note on Workflow:**
> For standard End-to-End ML pre-flight steps (Data Ingestion, Missing Values, Train/Test Split), refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document focuses solely on the **foundational mental model unique to K-Nearest Neighbors (KNN)**.

---

## 1. What Makes KNN Fundamentally Different?

In all parametric and linear models (Linear/Logistic Regression, Ridge, Lasso, ElasticNet, SGD):
$$\hat{y} = f(\mathbf{x}; \mathbf{w}, b) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
The model optimizes and learns a fixed set of weights $\mathbf{w}$ and intercept $b$ during `.fit()`. Once weights are computed, **the original training data can be thrown away**. At inference time, predicting on a new point takes just a simple dot product: $O(D)$ time where $D$ is the number of features.

**K-Nearest Neighbors completely inverts this paradigm:**

```
Linear Models (Eager Learning):
[Training Data] ──fit()──> [Learns Parameters w, b] ──> Discard Training Data!
[New Query Point x*] ──predict()──> Compute w^T x* + b (Instant O(D))

KNN (Lazy / Instance-Based Learning):
[Training Data] ──fit()──> [Store All Data in Memory (O(1) Training!)]
[New Query Point x*] ──predict()──> Scan All Training Points, Calculate Distances, Sort, Vote! (Expensive O(N*D))
```

---

## 2. "Eager" vs "Lazy" Learning Comparison

| Dimension | Eager Learners (Logistic / SGD / Trees) | Lazy Learner (KNN) |
| :--- | :--- | :--- |
| **Training Time Complexity** | $O(N \cdot D)$ or $O(N \cdot D \cdot \text{epochs})$ (Heavy) | $O(1)$ — literally just stores pointers to dataset in memory |
| **Inference Time Complexity** | $O(D)$ (Instantaneous dot product) | $O(N \cdot D)$ for brute force (Slow for large $N$) |
| **Model Storage Size** | Minimal: just weights array $\mathbf{w} \in \mathbb{R}^D$ (few KB) | Massive: Entire dataset $X_{\text{train}} \in \mathbb{R}^{N \times D}$ (MBs to GBs) |
| **Adaptability to New Data** | Requires retraining or incremental `partial_fit` | Zero cost: Just append new row to dataset! |
| **Assumptions about Data** | Assumes linear separability or specific functional form | **Zero assumptions** (Non-parametric) |

---

## 3. Parametric vs Non-Parametric

1. **Parametric Models:**
   * Assume that the true data-generating process fits a fixed mathematical equation with a fixed number of parameters (e.g., $D$ weights).
   * Fast at test time, but if the assumption is wrong (e.g. data is circular/spiral), the model suffers from high bias (underfitting).

2. **Non-Parametric Models (KNN):**
   * Does NOT mean "zero parameters". It means the **number of effective parameters is not fixed in advance**; it grows with the size of the training dataset.
   * KNN can approximate **arbitrary, highly non-linear, disjoint decision boundaries** (e.g., concentric circles, islands, spirals) without feature transformations or polynomial features.

---

## 4. The Core Voting Mechanism

Given a query point $\mathbf{x}^*$:
1. Calculate distance $d(\mathbf{x}^*, \mathbf{x}_i)$ to every training point $\mathbf{x}_i \in X_{\text{train}}$.
2. Find the $K$ points with the smallest distances: $\mathcal{N}_K(\mathbf{x}^*)$.
3. **Classification Rule:**
   $$\hat{y} = \arg\max_{c \in \{0, 1, \dots, C-1\}} \sum_{i \in \mathcal{N}_K(\mathbf{x}^*)} \mathbb{I}(y_i = c)$$
   *(Assign the class held by the majority of the $K$ neighbors).*

```
       (+)         (+)
              x* (?)     (-)
       (+)         (-)
             (-)
If K=3: 2 are (+), 1 is (-)  ==> Predict (+)
If K=5: 2 are (+), 3 are (-)  ==> Predict (-)
```

> **Key Rule of Thumb:** For binary classification, always choose an **odd number for $K$** (e.g., $K=3, 5, 7, 11$) to prevent 50-50 tie breaks!
