# 🚀 Gradient Boosting Machine (GBM): Philosophy & Function-Space Gradient Descent

Jerome Friedman's 1999/2001 breakthrough unified boosting as **Gradient Descent in Function Space**.

---

## 1. Parameter-Space vs. Function-Space Optimization

In traditional optimization (e.g. Logistic Regression, Neural Networks), we minimize empirical risk $\mathcal{L}(\theta)$ by adjusting finite parameters $\theta \in \mathbb{R}^d$:
$$\theta^{(m)} = \theta^{(m-1)} - \eta \nabla_\theta \mathcal{L}(\theta)$$

In **Gradient Boosting**, we don't optimize parameters $\theta$. Instead, we optimize the **function itself** $F(x)$:

$$F_m(x) = F_{m-1}(x) - \eta \cdot g_m(x)$$

Where $g_m(x)$ is the gradient of the loss function with respect to the ensemble's current prediction evaluated at each training point:

$$g_{im} = \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$

Since $g_{im}$ is only defined on the discrete training points $\{x_i\}_{i=1}^N$, we **fit a base decision tree $h_m(x)$ to predict the negative gradient values $-g_{im}$**!
The decision tree acts as a continuous interpolator/generalizer of the gradient vector to unseen points in feature space.
