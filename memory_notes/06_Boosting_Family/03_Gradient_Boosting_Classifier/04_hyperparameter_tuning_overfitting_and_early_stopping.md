# 🛑 Overfitting Control & Early Stopping in Gradient Boosting

Because Gradient Boosting relentlessly fits residual errors, it will eventually memorize noise and overfit if trained for too many iterations without regularization.

---

## 1. The Early Stopping Mechanism

Instead of guessing the optimal `n_estimators`, use validation-based early stopping:

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier(
    n_estimators=1000,           # Set generously high
    learning_rate=0.05,
    max_depth=4,
    validation_fraction=0.15,    # 15% internal validation set
    n_iter_no_change=10,         # Stop if validation score stalls for 10 rounds
    tol=1e-4,
    random_state=42
)
gb_clf.fit(X_train, y_train)

print(f"Optimal iterations reached: {gb_clf.n_estimators_}")
```

---

## 2. Key Hyperparameters Checklist

| Hyperparameter | Typical Sweet Spot | Role |
| :--- | :--- | :--- |
| `learning_rate` | $0.03 - 0.1$ | Step size shrinkage. Lower is always better if compute allows. |
| `max_depth` | $3 - 6$ | Interaction depth. Unlike RF ($d \sim 15+$), GBM uses shallow trees! |
| `subsample` | $0.7 - 0.85$ | Row subsampling fraction. |
| `max_features` | `'sqrt'` or $0.8$ | Column subsampling fraction. |
| `min_samples_leaf` | $10 - 50$ | Leaf smoothing; prevents isolating rare outliers. |
