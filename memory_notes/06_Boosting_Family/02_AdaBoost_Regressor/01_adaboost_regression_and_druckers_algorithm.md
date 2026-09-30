# 📉 AdaBoost Regression: Drucker's Algorithm (AdaBoost.R2)

AdaBoost for continuous target variables was formulated by Harris Drucker (1997) as the **AdaBoost.R2** algorithm.

---

## 1. Core Mechanics of Drucker's Algorithm

1. **Step 1: Compute Relative Losses $L_i$:**
   At stage $m$, evaluate base regressor $h_m(x_i)$ and compute the maximum prediction error:
   $$D = \max_{i=1}^N |y_i - h_m(x_i)|$$
   Normalize each observation's error to the range $[0, 1]$ using a chosen loss function:
   - **Linear:** $L_i = \frac{|y_i - h_m(x_i)|}{D}$
   - **Square:** $L_i = \frac{(y_i - h_m(x_i))^2}{D^2}$
   - **Exponential:** $L_i = 1 - \exp\Big( -\frac{|y_i - h_m(x_i)|}{D} \Big)$

2. **Step 2: Average Loss & Estimator Confidence $\beta_m$:**
   $$\bar{L}_m = \sum_{i=1}^N w_i^{(m)} L_i$$
   $$\beta_m = \frac{\bar{L}_m}{1 - \bar{L}_m}$$
   The stage weight assigned to $h_m$ is $\alpha_m = \ln(1 / \beta_m)$.

3. **Step 3: Update Sample Weights:**
   $$w_i^{(m+1)} = w_i^{(m)} \beta_m^{1 - L_i}$$
   Renormalize so $\sum w_i^{(m+1)} = 1$.

---

## 2. Prediction via Weighted Median

Unlike standard regression ensembles which compute a simple weighted arithmetic mean $\frac{\sum \alpha_m h_m}{\sum \alpha_m}$, Drucker's AdaBoost.R2 calculates the **Weighted Median** of the base predictions:

$$\hat{y}_{\text{final}}(x) = \text{Weighted-Median}\Big( \{h_m(x)\}_{m=1}^M, \{\ln(1/\beta_m)\}_{m=1}^M \Big)$$

- The weighted median provides superior mathematical robustness against extreme outlier predictions from individual base trees!
