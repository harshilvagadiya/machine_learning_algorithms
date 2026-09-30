# 🎛️ Hyperparameter Optimization & Shrinkage in AdaBoost.R2

Tuning `AdaBoostRegressor` involves orchestrating the interaction between base tree complexity, shrinkage rate $\eta$, and loss metric.

---

## 1. Key Hyperparameters

| Parameter | Recommended Range | Impact on Model |
| :--- | :--- | :--- |
| `n_estimators` | $50 - 300$ | Capacity to model complex trends. Higher values can overfit if $\eta$ is large. |
| `learning_rate` ($\eta$) | $0.01 - 0.20$ | Shrinks each tree's impact; lower values require higher `n_estimators`. |
| `estimator__max_depth` | $3 - 6$ | Unlike classification stumps ($d=1$), regression boosting benefits from shallow trees ($d \approx 4-6$) to capture interactions. |
| `loss` | `'linear'`, `'square'`, `'exponential'` | Adapts to error distribution and outlier presence. |

---

## 2. Production GridSearchCV Recipe

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'loss': ['linear', 'square'],
    'estimator__max_depth': [3, 5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    AdaBoostRegressor(estimator=DecisionTreeRegressor(), random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1
)
grid.fit(X_train, y_train)
```
