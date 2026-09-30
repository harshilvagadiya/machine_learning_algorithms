# ⚙️ HistGradientBoosting Classifier & Regressor Mechanics

`HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` are drop-in replacements for standard Gradient Boosting with $10\times - 100\times$ faster training throughput.

---

## 1. Algorithmic Highlights

- **Loss Functions:**
  - Classifier: `loss='log_loss'` (binary / multiclass cross-entropy).
  - Regressor: `loss='squared_error'`, `'absolute_error'`, `'gamma'`, `'poisson'`.
- **Automatic Early Stopping:**
  - Enabled by default (`early_stopping='auto'`).
  - Automatically reserves 10% of training data for validation and halts boosting if score doesn't improve for `n_iter_no_change=10` iterations.
- **Monotonic Constraints:**
  - Supports strict monotonic constraints (`monotonic_cst=[+1, -1, 0, ...]`) ensuring business rules (e.g. higher credit score never increases default probability).

---

## 2. Minimalist Enterprise Implementation

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Automatically handles NaNs and categorical columns natively!
hgb_clf = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.08,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    categorical_features=categorical_col_indices,
    random_state=42
)
hgb_clf.fit(X_train, y_train)
```
