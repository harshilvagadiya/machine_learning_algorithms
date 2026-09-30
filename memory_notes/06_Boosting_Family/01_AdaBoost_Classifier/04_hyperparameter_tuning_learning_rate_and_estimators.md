# 🎛️ Hyperparameter Tuning: Learning Rate & Estimators in AdaBoost

AdaBoost performance is governed primarily by the interaction between the number of base estimators ($M$) and the learning rate ($\eta$).

---

## 1. The Shrinkage Principle (Learning Rate $\eta$)

With shrinkage introduced by Friedman (2001), the ensemble update becomes:

$$F_m(x) = F_{m-1}(x) + \eta \cdot \alpha_m h_m(x), \quad 0 < \eta \le 1$$

- When $\eta < 1$, each weak learner contributes a shrunken update.
- This forces the ensemble to take smaller, more cautious steps in function space, significantly reducing test error and preventing premature overfitting.
- **Rule of Thumb:** Decreasing $\eta$ requires a proportional increase in `n_estimators`.

---

## 2. Hyperparameter Grid & Tuning Strategy

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2]  # Stumps (d=1) or depth-2 trees
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
grid.fit(X_train, y_train)
```
