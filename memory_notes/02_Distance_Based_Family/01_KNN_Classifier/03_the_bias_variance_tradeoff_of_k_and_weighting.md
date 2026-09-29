# Guide 03: The Bias-Variance Tradeoff of $K$ & Neighbor Weighting

> **Note on Workflow:**
> For standard cross-validation loops, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **hyperparameter mechanics of $K$ and distance-weighted voting**.

---

## 1. The Critical Hyperparameter: $K$

The choice of $K$ is the central regularizer in KNN. Unlike parametric models where regularization is $\lambda$ or $\alpha$, in KNN **$K$ directly controls model complexity**:

```
Small K (e.g. K=1) ──────────────────────────────────────────> Large K (e.g. K=N)
Complex Decision Boundary                                    Linear / Flat Boundary
High Variance / Low Bias                                     Low Variance / High Bias
Overfitting Risk (Noise Memorization)                        Underfitting Risk (Global Mode)
```

### Detailed Breakdown:

| Parameter Value | Model Behavior | Decision Boundary | Train Error | Test Error |
| :--- | :--- | :--- | :--- | :--- |
| **$K = 1$** | Each point defines its own territory (Voronoi cell). A single noisy mislabeled point creates an isolated island. | Highly jagged, fragmented, erratic | **0.0%** (Memorizes training set) | High (Overfitting) |
| **$K \approx \sqrt{N}$ (Odd)** | Balances local neighborhood context with statistical smoothing. | Smooth, resilient to individual point jitter | Moderate | **Optimal** (Lowest error) |
| **$K \to N$** | Every query point consults the entire training set. The majority class always wins everywhere. | Completely flat / null boundary | High | High (Underfitting) |

---

## 2. Neighbor Weighting: Uniform vs Distance

In standard KNN (`weights='uniform'`), all $K$ neighbors have an **equal 1 vote**, regardless of whether a neighbor is $0.001$ units away or $100$ units away.

### 2.1 The Problem with Uniform Voting:
Imagine $K=5$, classifying a new point $\mathbf{x}^*$:
* Neighbor 1: distance = $0.02$, Class = `Fraud`
* Neighbor 2: distance = $0.03$, Class = `Fraud`
* Neighbor 3: distance = $4.80$, Class = `Normal`
* Neighbor 4: distance = $4.90$, Class = `Normal`
* Neighbor 5: distance = $5.00$, Class = `Normal`

With `weights='uniform'`: 3 votes for `Normal` vs 2 votes for `Fraud` $\implies$ **Predicts `Normal` (Wrong!)**.
Even though the query point is practically sitting on top of two `Fraud` points, the distant `Normal` points overpowered it!

---

### 2.2 Distance Weighting (`weights='distance'`)
Each neighbor's vote is weighted inversely proportional to its distance:
$$w_i = \frac{1}{d(\mathbf{x}^*, \mathbf{x}_i)}$$

Class probability for class $c$:
$$P(y = c \mid \mathbf{x}^*) = \frac{\sum_{i \in \mathcal{N}_K, y_i = c} w_i}{\sum_{i \in \mathcal{N}_K} w_i}$$

* In our example above:
  * $w_1 = \frac{1}{0.02} = 50.0$, $w_2 = \frac{1}{0.03} = 33.3$
  * Sum Fraud weights = $83.3$
  * Sum Normal weights = $0.612$
  * **Result:** `Fraud` wins decisively with $> 99\%$ weighted probability!

### 2.3 When to Use Which?
* **`weights='uniform'`:** Best when data density is uniform and labels are relatively clean.
* **`weights='distance'`:** Best when data has varying density, class imbalance, or transition zones between sparse and dense clusters.
