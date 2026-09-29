# Guide 06: Hyperparameter Tuning & Non-Linear Decision Boundaries

> **Note on Workflow:**
> For standard evaluation metrics (Confusion Matrix, ROC-AUC, Classification Report), refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **grid search strategy and non-linear decision boundary properties of KNN**.

---

## 1. The Optimal Hyperparameter Grid

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 13, 17, 21, 29],  # Odd values to prevent ties
    'knn__weights': ['uniform', 'distance'],            # Equal vs inverse-distance votes
    'knn__metric': ['minkowski'],
    'knn__p': [1, 2],                                   # 1=Manhattan, 2=Euclidean
    'knn__leaf_size': [20, 30, 50]
}
```

---

## 2. Non-Linear Decision Boundaries (KNN's Superpower)

Linear models (Logistic Regression, Linear SVM) can only draw straight lines ($w_1 x_1 + w_2 x_2 + b = 0$). They completely fail on:
1. **Concentric Circles:** Inner circle is Class 0, outer ring is Class 1.
2. **Moons / Interlocking Spirals:** Two interlocking crescent shapes.
3. **The XOR Problem:** Opposite quadrants share the same label.

**KNN solves all of these out of the box with ZERO polynomial transformations:**
Because KNN is purely local, its decision boundary is formed by the union of local neighborhoods:
* At $K=1$, boundaries are exact **Voronoi polygons**.
* As $K$ increases, boundaries become smooth, organic, non-linear contours that seamlessly flow around complex clusters.
