# 🟡 Distance-Based Family: K-Nearest Neighbors (KNN) Classifier

Welcome to the **K-Nearest Neighbors (KNN) Classifier** architectural knowledge hub.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard end-to-end ML lifecycle steps (Data ingestion, missing value imputation, EDA, stratified train/test splitting, confusion matrices, and ROC curves) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> **The guides below document ONLY the novel mathematical, geometric, and operational concepts unique to Distance-Based Classification.**

---

## 📚 Complete KNN Classifier Guide Suite

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                   DISTANCE-BASED CLASSIFICATION MAP                   │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Foundational Theory                                    Engineering & Systems
   ├── 01. The Lazy Learner & Non-Parametric Intuition    ├── 04. Curse of Dimensionality & Mitigation
   ├── 02. Distance Metrics & Geometric Math              ├── 05. Search Algos (Brute vs KD vs Ball)
   └── 03. Bias-Variance Tradeoff of K & Weighting        ├── 06. Hyperparameter Tuning & Non-Linear Boundary
                                                          └── 07. Production Pipeline, SLAs & Modern ANN
```

| Guide | Core Focus | Key Questions Answered |
| :--- | :--- | :--- |
| **[01. The Lazy Learner & Intuition](01_the_lazy_learner_and_non_parametric_intuition.md)** | Non-parametric mental model | Why is training $O(1)$ and inference $O(N \cdot D)$? What does lazy learning mean? |
| **[02. Distance Metrics & Geometry](02_distance_metrics_and_geometric_math.md)** | Minkowski, Euclidean, Manhattan | Why does unscaled data destroy KNN completely? When to use Manhattan vs Euclidean? |
| **[03. Bias-Variance & Weighting](03_the_bias_variance_tradeoff_of_k_and_weighting.md)** | Regularization via $K$ | What happens at $K=1$ vs $K=N$? How does `weights='distance'` prevent majority tyranny? |
| **[04. Curse of Dimensionality](04_curse_of_dimensionality_and_mitigation.md)** | Geometric collapse in high-D | Why do all points become equidistant as $D \to \infty$? How does PCA save KNN? |
| **[05. Search Algorithms](05_search_algorithms_brute_vs_kdtree_vs_balltree.md)** | Fast spatial indexing | How do KD-Tree and Ball-Tree prune searches? Why does KD-tree fail when $D > 20$? |
| **[06. Hyperparameter Tuning](06_hyperparameter_tuning_and_decision_boundaries.md)** | Optimization & Complex Shapes | How does KNN effortlessly classify non-linear concentric circles and XOR problems? |
| **[07. Production & Modern ANN](07_production_pipeline_latency_and_ann.md)** | Deployment & Scale | Why is model size equal to dataset size? When to migrate to Faiss / HNSW? |

---

## ⚡ The KNN Classification Cheat Sheet

* **Classification Rule:** $\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_K} w_i \mathbb{I}(y_i = c)$
* **Feature Scaling:** **MANDATORY** (`StandardScaler`, `MinMaxScaler`, or `RobustScaler`).
* **Choosing $K$:** Start with $K \approx \sqrt{N}$ (choose an odd number for binary classification).
* **Weights:** Use `weights='distance'` if density is non-uniform or minority class is clustered tightly.
* **Metric:** Use `p=2` (Euclidean) for low dimensions ($D \le 15$); use `p=1` (Manhattan) or prepend PCA when $D > 20$.
