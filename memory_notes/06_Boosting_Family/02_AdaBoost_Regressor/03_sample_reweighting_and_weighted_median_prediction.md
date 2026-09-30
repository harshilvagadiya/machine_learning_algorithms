# ⚖️ Sample Reweighting & Weighted Median Prediction in AdaBoost.R2

Here we examine the mathematical intuition of how sample weights adjust in regression boosting and how the final continuous prediction is aggregated.

---

## 1. Sample Reweighting Dynamics

The weight update rule:
$$w_i^{(m+1)} = w_i^{(m)} \beta_m^{1 - L_i}$$

Recall that $\beta_m = \frac{\bar{L}_m}{1 - \bar{L}_m} \in (0, 1)$ for any learner better than random.
- For a sample with **zero error** ($L_i = 0$):
  $$w_i^{(m+1)} = w_i^{(m)} \beta_m^1 < w_i^{(m)}$$
  (Weight is reduced by factor $\beta_m$).
- For a sample with **maximum error** ($L_i = 1$):
  $$w_i^{(m+1)} = w_i^{(m)} \beta_m^0 = w_i^{(m)}$$
  (Weight remains unchanged, thus increasing in relative normalized proportion!).

---

## 2. Weighted Median Calculation

To calculate the weighted median of predictions $h_1(x), \dots, h_M(x)$ with weights $\alpha_m = \ln(1 / \beta_m)$:

1. Sort the predictions in ascending order:
   $$h_{(1)}(x) \le h_{(2)}(x) \le \dots \le h_{(M)}(x)$$
2. Associate corresponding weights $\alpha_{(1)}, \dots, \alpha_{(M)}$.
3. Find the smallest index $k$ such that:
   $$\sum_{j=1}^k \alpha_{(j)} \ge \frac{1}{2} \sum_{m=1}^M \alpha_m$$
4. Set $\hat{y}_{\text{final}}(x) = h_{(k)}(x)$.

This ensures that the final prediction is completely insulated from any single runaway base learner prediction.
