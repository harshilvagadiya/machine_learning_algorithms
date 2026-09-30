# 📉 Gradient Boosting Regression: Philosophy & Loss Functions

Gradient Boosting for regression fits shallow decision trees sequentially to the negative gradients (pseudo-residuals) of a chosen differentiable regression loss function.

---

## 1. Supported Loss Functions

Scikit-learn's `GradientBoostingRegressor(loss='squared_error'|'absolute_error'|'huber'|'quantile')` supports four distinct objectives:

| Loss Function | Mathematical Objective $L(y, F)$ | Pseudo-Residual $- \frac{\partial L}{\partial F}$ | Sensitivity to Outliers |
| :--- | :--- | :--- | :--- |
| **Squared Error** (`'squared_error'`) | $\frac{1}{2}(y - F)^2$ | $y - F$ (True standard residual) | High (Quadratic penalty) |
| **Absolute Error** (`'absolute_error'`) | $|y - F|$ | $\text{sign}(y - F)$ | Low (Robust median estimation) |
| **Huber Loss** (`'huber'`) | Quadratic for $|y-F| \le \delta$, Linear for $|y-F| > \delta$ | Capped residuals at $\pm \delta$ | **Balanced** (Smooth yet robust) |
| **Quantile Loss** (`'quantile'`) | Pinball loss at quantile $\alpha$ | $\alpha$ if $y > F$, else $\alpha - 1$ | Unbiased conditional quantile estimation |

---

## 2. Squared Error Special Case: Residual Chasing

Under Squared Error Loss, the negative gradient is identically equal to the raw error residual:

$$r_{im} = -\left[ \frac{\partial \frac{1}{2}(y_i - F(x_i))^2}{\partial F(x_i)} \right] = y_i - F_{m-1}(x_i)$$

Thus, gradient boosting under squared error is simply **iterative residual fitting**:
$$\hat{y}_0 = \bar{y}$$
$$\text{Residual}_1 = y - \hat{y}_0 \implies \text{Fit } h_1(x)$$
$$\text{Residual}_2 = y - (\hat{y}_0 + \eta h_1(x)) \implies \text{Fit } h_2(x)$$
This gives clear, concrete intuition to the function-space gradient descent process!
