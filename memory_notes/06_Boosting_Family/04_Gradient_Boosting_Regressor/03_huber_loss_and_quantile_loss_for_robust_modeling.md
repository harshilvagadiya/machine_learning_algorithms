# 🛡️ Huber Loss & Quantile Loss for Robust Regression Modeling

Real-world enterprise datasets frequently contain corrupt sensor anomalies, unannounced accounting changes, or extreme fat-tailed continuous targets. Standard MSE gradient boosting fails by aggressively distorting leaf values towards these outliers.

---

## 1. Huber Loss (`loss='huber'`)

Huber loss acts quadratically for small errors and linearly for large errors, parameterized by threshold $\delta$:

$$L_\delta(y, F) = \begin{cases} \frac{1}{2}(y - F)^2 & \text{if } |y - F| \le \delta \\ \delta |y - F| - \frac{1}{2}\delta^2 & \text{if } |y - F| > \delta \end{cases}$$

### Negative Gradient (Pseudo-Residual):
$$r = -\frac{\partial L}{\partial F} = \begin{cases} y - F & \text{if } |y - F| \le \delta \\ \delta \cdot \text{sign}(y - F) & \text{if } |y - F| > \delta \end{cases}$$

- Outliers are capped at $\pm \delta$! They can never disproportionately pull leaf values or distort tree split criteria.
- In scikit-learn, $\delta$ is automatically set to the $\alpha$-quantile of absolute residuals (`alpha=0.9` by default).

---

## 2. Quantile Regression (`loss='quantile'`)

To predict specific conditional percentiles (e.g. 10th percentile worst-case risk, 90th percentile peak capacity):

```python
from sklearn.ensemble import GradientBoostingRegressor

# Train models for Lower (10%), Median (50%), and Upper (90%) Prediction Intervals
lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.10, random_state=42).fit(X_train, y_train)
median_model = GradientBoostingRegressor(loss='quantile', alpha=0.50, random_state=42).fit(X_train, y_train)
upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.90, random_state=42).fit(X_train, y_train)

# Generates calibrated 80% prediction intervals [y_lower, y_upper]
y_lower = lower_model.predict(X_test)
y_median = median_model.predict(X_test)
y_upper = upper_model.predict(X_test)
```
