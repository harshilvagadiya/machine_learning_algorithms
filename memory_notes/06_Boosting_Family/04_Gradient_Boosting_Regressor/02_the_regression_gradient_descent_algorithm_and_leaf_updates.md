# 📐 The Regression Gradient Descent Algorithm & Leaf Updates

Here is the exact mathematical formulation of Jerome Friedman's Gradient Boosting Regressor algorithm.

---

## 1. Algorithm Derivation

### Step 1: Initialize Constant Optimal Model
$$F_0(x) = \arg\min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)$$
- For squared error: $F_0(x) = \bar{y}$ (sample mean).
- For absolute error: $F_0(x) = \text{median}(y)$ (sample median).

### Step 2: Boosting Loop for $m = 1, 2, \dots, M$
1. **Compute Pseudo-Residuals:**
   $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}, \quad i = 1, \dots, N$$

2. **Fit Regression Tree:**
   Train a regression tree to predict $\{r_{im}\}_{i=1}^N$ from $\{x_i\}_{i=1}^N$, yielding terminal regions $R_{jm}$ for $j = 1, \dots, J_m$.

3. **Compute Optimal Leaf Output Value $\gamma_{jm}$:**
   $$\gamma_{jm} = \arg\min_{\gamma} \sum_{x_i \in R_{jm}} L\big(y_i, F_{m-1}(x_i) + \gamma\big)$$
   - For squared error: $\gamma_{jm} = \frac{1}{|R_{jm}|} \sum_{x_i \in R_{jm}} r_{im}$ (average residual in the leaf).
   - For absolute error: $\gamma_{jm} = \text{median}_{x_i \in R_{jm}} (y_i - F_{m-1}(x_i))$.

4. **Update the Ensemble with Shrinkage $\eta$:**
   $$F_m(x) = F_{m-1}(x) + \eta \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(x \in R_{jm})$$

---

## 2. Final Prediction
$$\hat{y}(x) = F_M(x) = F_0(x) + \eta \sum_{m=1}^M \sum_{j=1}^{J_m} \gamma_{jm} \mathbb{I}(x \in R_{jm})$$
