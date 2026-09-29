# 🟣 Tree-Based Family: Decision Tree Classifier

Welcome to the **Decision Tree Classifier** architectural knowledge hub.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard end-to-end ML lifecycle steps (Data ingestion, missing value imputation, EDA, train/test splitting, ColumnTransformer preprocessing, and evaluation metrics) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> **The guides below document ONLY the novel mathematical, algorithmic, and operational concepts unique to Decision Trees.**

---

## 📚 Complete Decision Tree Guide Suite

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    DECISION TREE CLASSIFICATION MAP                    │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Foundational Theory                                    Engineering & Regularization
   ├── 01. Recursive Binary Partitioning                  ├── 04. Minimal Cost-Complexity Pruning (ccp)
   ├── 02. Gini Impurity vs Shannon Entropy               ├── 05. MDI vs Permutation Feature Importance
   └── 03. The Overfitting Monster & Pre-Pruning          ├── 06. Tree Visualization & Decision Surfaces
                                                          └── 07. Production Pipelines & Latency SLAs
```

| Guide | Core Focus | Key Questions Answered |
| :--- | :--- | :--- |
| **[01. Recursive Partitioning](01_the_recursive_partitioning_and_splitting_intuition.md)** | Geometric Slicing | Axis-aligned hyperplanes, greedy CART algorithm, and scale-invariance superpower. |
| **[02. Gini vs. Entropy](02_splitting_criteria_entropy_information_gain_vs_gini_impurity.md)** | Splitting Criteria | Gini vs Information Gain math, impurity definitions, and why Gini dominates production. |
| **[03. Overfitting & Pre-Pruning](03_the_overfitting_monster_and_pre_pruning_hyperparameters.md)** | Depth Regularization | Why unconstrained trees memorize noise, and tuning `max_depth` vs `min_samples_leaf`. |
| **[04. Cost-Complexity Pruning](04_post_pruning_cost_complexity_pruning_ccp_alpha.md)** | Post-Pruning Theory | Objective function $R_\alpha(T) = R(T) + \alpha |T|$, and finding optimal `ccp_alpha`. |
| **[05. Feature Importance](05_feature_importance_impurity_decrease_vs_permutation.md)** | Importance Diagnostics | The high-cardinality MDI bias trap, and why Permutation Importance solves it. |
| **[06. Tree Visualization](06_tree_visualization_and_decision_surfaces.md)** | Visual & Text Inspection | `plot_tree`, `export_text`, and the diagonal staircase failure trap. |
| **[07. Production & Latency](07_production_pipeline_tree_latency_and_deployment.md)** | Deployment & Speed | Sub-millisecond latency ($< 0.1\text{ms}$), `.joblib` serialization, and SQL export. |

---

## ⚡ The Decision Tree Cheat Sheet

* **Splitting Criteria:** Gini ($1 - \sum p_k^2$) vs Entropy ($-\sum p_k \log_2 p_k$).
* **Feature Scaling:** **NOT REQUIRED** (Monotonic transformations preserve split ordering).
* **The Overfitting Shield:** Never deploy with `max_depth=None`. Set `max_depth` and `min_samples_leaf >= 5`.
* **Interpretability:** Extract direct human-readable boolean if-else rules via `export_text`.
* **Traversal Speed:** Pure pointer hops down tree depth — lightning fast ($< 0.1\text{ms}$).

---

## 🧪 Enterprise Production Notebooks Suite (4 Executed Implementations)

The companion practical notebooks in [`supervised_learning/DecisionTreeClassifier/`](file:///home/python/03_Custom_addons/ML/supervised_learning/DecisionTreeClassifier/) implement real-world end-to-end applications across diverse domains:

1. **[Heart Disease Clinical Diagnostic (DecisionTreeClassifier)](file:///home/python/03_Custom_addons/ML/supervised_learning/DecisionTreeClassifier/Heart_Disease_Diagnostic_DecisionTreeClassifier.ipynb)**
   * *Domain:* Clinical Cardiology & Diagnostic Rule Induction
   * *Model:* `heart_disease_tree_pipeline.joblib`
2. **[Credit Card Default Risk Assessment (DecisionTreeClassifier)](file:///home/python/03_Custom_addons/ML/supervised_learning/DecisionTreeClassifier/Credit_Default_Risk_DecisionTreeClassifier.ipynb)**
   * *Domain:* Banking Credit Approval & Lending Governance
   * *Model:* `credit_default_tree_pipeline.joblib`
3. **[Employee Turnover & Attrition Prediction (DecisionTreeClassifier)](file:///home/python/03_Custom_addons/ML/supervised_learning/DecisionTreeClassifier/Employee_Attrition_DecisionTreeClassifier.ipynb)**
   * *Domain:* Human Capital Analytics & Retention Policy
   * *Model:* `employee_attrition_tree_pipeline.joblib`
4. **[Iris Species Multiclass Classification (DecisionTreeClassifier)](file:///home/python/03_Custom_addons/ML/supervised_learning/DecisionTreeClassifier/Iris_Multiclass_DecisionTreeClassifier.ipynb)**
   * *Domain:* Botanical Phenotype & Multiclass Tree Visualization
   * *Model:* `iris_multiclass_tree_pipeline.joblib`
