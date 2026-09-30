# ⛓️ Multi-Output & Chain Architecture (MOC, CC, MOR, RC) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Real-world enterprise systems frequently require predicting **multiple simultaneous targets** $\mathbf{y} = [y_1, y_2, \dots, y_m]^T$ from input features $\mathbf{x} \in \mathbb{R}^p$:
- Medical Diagnostics: Predicting presence of diabetes, hypertension, and heart disease simultaneously.
- Quantitative Finance: Jointly forecasting stock return, trading volume, and volatility.
- Autonomous Robotics: Predicting steering angle, throttle percentage, and braking pressure in tandem.

There are two primary architectural paradigms for multi-output problems:
1. **Independent Estimators (`MultiOutputClassifier`, `MultiOutputRegressor`):** Decomposes the multivariate problem into $m$ independent single-target models.
2. **Autoregressive Conditional Chains (`ClassifierChain`, `RegressorChain`):** Models the joint target distribution by chaining predictions sequentially, allowing earlier predicted targets to serve as augmented input features for subsequent targets.

---

## 2. MultiOutput vs Chains Comparison

| Dimension | MultiOutput (MOC / MOR) | Chains (ClassifierChain / RegressorChain) |
| :--- | :--- | :--- |
| **Target Independence** | Assumes $P(\mathbf{y} \mid \mathbf{x}) = \prod_{j=1}^m P(y_j \mid \mathbf{x})$ | Captures cross-target conditional dependencies |
| **Model Structure** | $m$ independent base estimators | $m$ sequential estimators where $f_j: (\mathbf{x}, \hat{y}_1, \dots, \hat{y}_{j-1}) \to y_j$ |
| **Order Sensitivity** | Invariant to target order | Sensitive to chain order (Mitigated via `order='random'` or Ensembles) |
| **Serving Latency** | Parallelizable across $m$ cores | Strictly sequential $m$-step dependency graph |
| **Hamming Loss / Metric** | Baseline multi-label performance | Superior exact match ratio and cross-label correlation modeling |

---

## 3. Mathematical Formulation of Chains

By the probability chain rule, the joint distribution of targets factorizes as:
$$P(y_1, y_2, \dots, y_m \mid \mathbf{x}) = P(y_1 \mid \mathbf{x}) \prod_{j=2}^m P(y_j \mid \mathbf{x}, y_1, \dots, y_{j-1})$$

In a **Classifier Chain** (Read et al., 2011), each base classifier $C_j$ is trained on an expanded feature vector:
$$\mathbf{z}_j = [\mathbf{x}^T, y_1, y_2, \dots, y_{j-1}]^T \in \mathbb{R}^{p + j - 1}$$
At test time, the chain recursively propagates predicted labels:
$$\hat{y}_j = C_j([\mathbf{x}^T, \hat{y}_1, \dots, \hat{y}_{j-1}]^T)$$

In a **Regressor Chain** (Spyromitros-Xioufis et al., 2016), continuous predictions $\hat{y}_j \in \mathbb{R}$ are propagated sequentially, allowing subsequent regression models to incorporate cross-target physical or economic constraints.

---

## 4. Production Engineering & Serving Latency
- **Exact Match vs Hamming Loss:** Independent models minimize Hamming Loss (per-label error) but suffer high 0-1 subset loss because they predict combinations that never occur in nature. Chains drastically improve Subset Accuracy.
- **Serving Overhead & Latency:**
  - MultiOutput: $\mathcal{O}(m \cdot \text{Cost}(f))$ executed in parallel ($\approx 1.2 \text{ ms}$).
  - Chains: $\mathcal{O}(m \cdot \text{Cost}(f))$ strictly sequential ($\approx 3.5 \text{ ms}$).
  - Both comfortably pass the enterprise **Sub-50ms SLA**.
- **Shared Preprocessor Pipeline:** By bundling ColumnTransformer once inside the meta-pipeline, feature extraction occurs exactly once per inference request, yielding a $2\times$ throughput increase over fragmented isolated models.
