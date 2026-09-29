# 03. The Overfitting Monster & Pre-Pruning Hyperparameters

Welcome to Guide 03 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **the mechanics of tree overfitting, the bias-variance tradeoff of depth, and pre-pruning regularization controls**.

---

## 1. The Overfitting Monster: 100% Train Accuracy Disaster

A decision tree left unconstrained (`max_depth=None`, `min_samples_split=2`) will split relentlessly until:
1. Every leaf contains exactly $1$ training sample, OR
2. Every leaf is $100\%$ pure.

```
       TRAINING ACCURACY: 100.00% (PURE MEMORIZATION)
       TEST ACCURACY    :  68.20% (CATASTROPHIC GENERALIZATION COLLAPSE!)
```

```
        TRAINING PERFORMANCE VS TREE DEPTH (THE OVERFITTING CURVE)
        
   Accuracy
     ▲                                   Train Accuracy (Memorization -> 100%)
     │                          .....................................
100% ┼                     . .
     │                  . .
 85% ┼                * * * * * *       <--- Optimal Sweet Spot (Depth = 4-6)
     │             . *             *
 70% ┼           .                   * *
     │         .                         * *  Test Accuracy (Generalization Drop)
     └────────┼──────────────┼──────────────┼──────────────┼─────► Tree Depth
             Depth 1        Depth 4        Depth 8       Depth 20+
          (High Bias)                    (High Variance / Overfitting)
```

An unconstrained tree does not learn generalizable patterns; it builds specialized narrow branches that isolate **training noise, mislabeled records, and statistical anomalies**.

---

## 2. Pre-Pruning Architecture: The Stopping Brakes

**Pre-pruning** halts tree construction early before leaves become overly pure and specialized.

In Scikit-Learn's `DecisionTreeClassifier`, pre-pruning is governed by five primary regularizers:

```
  ┌────────────────────────┬────────────────────────────────────────────────────────┐
  │ HYPERPARAMETER         │ ARCHITECTURAL ROLE & REGULARIZATION BEHAVIOR           │
  ├────────────────────────┼────────────────────────────────────────────────────────┤
  │ max_depth              │ Hard limit on tree depth (root is 0).                  │
  │                        │ Chief brake against high variance. Typical: [3, 8].    │
  ├────────────────────────┼────────────────────────────────────────────────────────┤
  │ min_samples_split      │ Minimum samples a node must contain to attempt split.  │
  │                        │ Default: 2. Setting to 10-50 prevents tiny splits.    │
  ├────────────────────────┼────────────────────────────────────────────────────────┤
  │ min_samples_leaf       │ Minimum samples GUARANTEED in every leaf node.         │
  │                        │ Most effective single parameter to smooth boundaries!  │
  ├────────────────────────┼────────────────────────────────────────────────────────┤
  │ max_leaf_nodes         │ Hard global budget on total terminal nodes in tree.    │
  │                        │ Grows tree in best-first manner based on purity drop.  │
  ├────────────────────────┼────────────────────────────────────────────────────────┤
  │ min_impurity_decrease  │ Node only splits if impurity drop Delta I >= threshold.│
  │                        │ Rejects splits that offer negligible statistical gain. │
  └────────────────────────┴────────────────────────────────────────────────────────┘
```

---

## 3. Deep Dive: `min_samples_leaf` vs. `min_samples_split`

Engineers frequently confuse these two parameters:

* `min_samples_split=10`: If a node has $12$ samples, it is allowed to split. However, it could split into a left child with $11$ samples and a right child with **$1$ sample** (isolating an outlier!).
* `min_samples_leaf=5`: The split is ONLY approved if **both** the left child AND right child receive at least $5$ samples. If a cut leaves $1$ sample on either side, it is **strictly forbidden**.

> **Senior Best Practice:**
> If you can only tune one parameter besides `max_depth`, tune **`min_samples_leaf`**. Setting `min_samples_leaf=5` or `10` immediately prevents the tree from creating outlier-isolated decision boundaries.

---

## 4. Production Grid Search Recipe

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 8, 10],
    'min_samples_leaf': [2, 5, 10, 20],
    'min_samples_split': [5, 10, 20, 50]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```
