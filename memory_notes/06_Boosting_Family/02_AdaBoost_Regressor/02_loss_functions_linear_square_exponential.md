# 📐 Loss Functions in AdaBoost.R2: Linear vs. Square vs. Exponential

The parameter `loss` in scikit-learn's `AdaBoostRegressor(loss='linear'|'square'|'exponential')` controls how absolute prediction errors $|y_i - \hat{y}_i|$ are transformed into the unit interval $[0, 1]$.

---

## 1. Mathematical Formulas & Curves

Let $e_i = |y_i - \hat{y}_i|$ and $D = \max_k e_k$.

| Loss Function | Formula $L_i(e_i)$ | Behavior on Small Errors | Behavior on Large Errors | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Linear** (Default) | $L_i = \frac{e_i}{D}$ | Proportional gradient | Moderate penalty | Standard continuous targets without extreme outliers |
| **Square** | $L_i = \frac{e_i^2}{D^2}$ | Very low penalty for small errors | Aggressive quadratic penalty | Precision physics/engineering where large errors are intolerable |
| **Exponential** | $L_i = 1 - \exp(-e_i / D)$ | Steep penalty near zero | Saturates near 1 for large errors | Data with heavy tails / noise where extreme errors shouldn't dominate |

---

## 2. Practical Tuning Example

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Testing all three loss functions
for loss in ['linear', 'square', 'exponential']:
    reg = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4),
        n_estimators=100,
        loss=loss,
        learning_rate=0.1,
        random_state=42
    )
    reg.fit(X_train, y_train)
    r2 = reg.score(X_test, y_test)
    print(f"Loss: {loss:12s} | Test R²: {r2*100:.2f}%")
```
