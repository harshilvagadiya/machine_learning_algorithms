# 📐 Mathematics of Pseudo-Residuals & Leaf Values in Gradient Boosting Classification

Here is the exact derivation of binary classification under Binomial Deviance (Log-Loss).

---

## 1. Binomial Deviance Loss

For binary labels $y_i \in \{0, 1\}$ and raw ensemble log-odds score $F(x) = \ln \big( \frac{p}{1 - p} \big)$:

$$p_i = \sigma(F(x_i)) = \frac{1}{1 + e^{-F(x_i)}}$$
$$L(y_i, F(x_i)) = - \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big] = - y_i F(x_i) + \ln(1 + e^{F(x_i)})$$

---

## 2. Deriving the Pseudo-Residuals

Taking the partial derivative with respect to $F(x_i)$:

$$\frac{\partial L}{\partial F(x_i)} = - y_i + \frac{e^{F(x_i)}}{1 + e^{F(x_i)}} = - y_i + p_i = -(y_i - p_i)$$

The negative gradient (pseudo-residual) is therefore:

$$r_{im} = -\left[ \frac{\partial L}{\partial F(x_i)} \right]_{F_{m-1}} = y_i - p_{i, m-1}$$

- For $y_i = 1$, residual is $1 - p_{i, m-1} > 0$.
- For $y_i = 0$, residual is $0 - p_{i, m-1} < 0$.

---

## 3. Leaf Node Output Values (Newton-Raphson Step)

A decision tree partitions the feature space into $J$ terminal leaves $\{R_{jm}\}_{j=1}^J$.
Because MSE was used to find tree splits, the average residual in leaf $j$ is not optimal under log-loss.
We solve for the optimal leaf value $\gamma_{jm}$ using a single Newton-Raphson step:

$$\gamma_{jm} = \frac{\sum_{x_i \in R_{jm}} r_{im}}{\sum_{x_i \in R_{jm}} p_{i, m-1}(1 - p_{i, m-1})}$$

The ensemble model update is:
$$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^J \gamma_{jm} \mathbb{I}(x \in R_{jm})$$
