# 🎛️ Hyperparameter Optimization in Gradient Boosting Regression

Tuning `GradientBoostingRegressor` balances step size, interaction depth, and stochastic decorrelation.

---

## 1. Hyperparameter Roles & Tuning Ranges

| Parameter | Recommended Range | Description |
| :--- | :--- | :--- |
| `learning_rate` ($\eta$) | $0.02 - 0.1$ | Shrinkage factor. Smaller values strictly enhance out-of-sample generalization. |
| `n_estimators` | $100 - 500$ | Number of sequential boosting stages. Pair with early stopping! |
| `max_depth` | $3 - 6$ | Maximum depth of individual regression trees (controls feature interaction order). |
| `subsample` | $0.7 - 0.85$ | Fraction of samples used per tree (introduces stochastic gradient descent). |
| `max_features` | `'sqrt'`, $0.8$, or $1.0$ | Number of features considered at each split. |
| `loss` | `'squared_error'`, `'huber'` | Choice of objective function. |

---

## 2. Production GridSearchCV Recipe

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.03, 0.08],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0],
    'loss': ['squared_error', 'huber']
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='r2',
    n_jobs=-1
)
grid.fit(X_train, y_train)
```
