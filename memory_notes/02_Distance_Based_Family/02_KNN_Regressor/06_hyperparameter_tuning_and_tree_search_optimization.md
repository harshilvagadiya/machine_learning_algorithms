# Guide 06: Hyperparameter Tuning & Spatial Tree Search

> **Note on Workflow:**
> For GridSearchCV pipelines, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details **hyperparameter grids and spatial search algorithms for regression**.

---

## 1. The Complete Hyperparameter Grid

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 13, 17, 21, 29],
    'knn__weights': ['uniform', 'distance'],            # Simple mean vs inverse distance
    'knn__metric': ['minkowski'],
    'knn__p': [1, 2],                                   # 1=Manhattan, 2=Euclidean
    'knn__leaf_size': [20, 30, 50]
}
```

---

## 2. Spatial Indexing Algorithms

In `KNeighborsRegressor(algorithm=...)`:
* **`'auto'` (Default):** Let scikit-learn pick the most efficient based on data shape.
* **`'kd_tree'`:** Builds axis-aligned hierarchical trees. Ultra-fast for $D < 20$.
* **`'ball_tree'`:** Groups points into nested hyperspheres using triangle inequality. Ideal for $D \\sim 20-50$ or specialized metrics.
* **`'brute'`:** Exhaustive $O(N \\cdot D)$ search. Recommended when $D > 50$ or data is very sparse.
