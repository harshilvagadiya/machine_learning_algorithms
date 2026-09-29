# 🟡 Distance-Based Family: K-Nearest Neighbors (KNN) Regressor

Welcome to the **K-Nearest Neighbors (KNN) Regressor** architectural knowledge hub.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard end-to-end ML lifecycle steps (Data ingestion, missing value imputation, EDA, train/test splitting, ColumnTransformer preprocessing, and residual metrics) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> **The guides below document ONLY the novel mathematical, geometric, and operational concepts unique to Distance-Based Regression.**

---

## 📚 Complete KNN Regressor Guide Suite

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                     DISTANCE-BASED REGRESSION MAP                      │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Foundational Theory                                    Engineering & Systems
   ├── 01. The Local Mean & Non-Parametric Intuition      ├── 04. Bias-Variance Tradeoff of K in Regression
   ├── 02. Uniform vs Distance-Weighted Surfaces          ├── 05. Regression Metrics & Residuals
   └── 03. The Extrapolation Disaster & Failure Modes     ├── 06. Hyperparameter Tuning & Spatial Trees
                                                          └── 07. Production Pipeline & Live Inference
```

| Guide | Core Focus | Key Questions Answered |
| :--- | :--- | :--- |
| **[01. The Local Mean & Intuition](01_the_local_mean_and_interpolation_intuition.md)** | Non-parametric Regression | How does KNN predict numbers without slope ($w$) or intercept ($b$)? Real estate appraisal analogy. |
| **[02. Uniform vs Distance Surfaces](02_uniform_vs_distance_weighted_surfaces.md)** | Surface Geometry | Why does `weights='uniform'` create a staircase step function? How does `weights='distance'` smooth it? |
| **[03. The Extrapolation Disaster](03_the_extrapolation_disaster_and_failure_modes.md)** | Critical Failure Modes | Why can KNN NEVER predict beyond $\\max(y_{\\text{train}})$? The 15,000 sq ft mansion disaster. |
| **[04. Bias-Variance of K](04_bias_variance_tradeoff_of_k_in_regression.md)** | Regularization via $K$ | What happens at $K=1$ ($R^2=1.0$, pure noise) vs $K=N$ (predicts global sample mean $\\bar{y}$)? |
| **[05. Metrics & Residuals](05_regression_evaluation_metrics_and_residuals.md)** | Diagnostics & Boundaries | How to diagnose boundary residual spikes (edge effect) and evaluate $R^2$, MAE, RMSE? |
| **[06. Hyperparameter Tuning](06_hyperparameter_tuning_and_tree_search_optimization.md)** | Spatial Tree Optimization | How to configure `GridSearchCV` across $K$, weights, metrics, and `leaf_size`? |
| **[07. Production & Deployment](07_production_pipeline_and_live_inference.md)** | Serialization & SLAs | How to package into a zero-leakage `.joblib` pipeline and ensure $< 10\\text{ms}$ live inference? |

---

## ⚡ The KNN Regression Cheat Sheet

* **Prediction Formula:** $\\hat{y} = \\frac{\\sum w_i y_i}{\\sum w_i}$ where $w_i = 1$ (uniform) or $w_i = \\frac{1}{d_i}$ (distance-weighted).
* **Feature Scaling:** **MANDATORY** (`StandardScaler`, `MinMaxScaler`, or `RobustScaler`).
* **Boundary Guard:** Predictions are strictly bounded: $\\min(y_{\\text{train}}) \\le \\hat{y} \\le \\max(y_{\\text{train}})$. Never use KNN for trend extrapolation!
* **Weights Recommendation:** Use `weights='distance'` for smooth, continuous regression surfaces.
