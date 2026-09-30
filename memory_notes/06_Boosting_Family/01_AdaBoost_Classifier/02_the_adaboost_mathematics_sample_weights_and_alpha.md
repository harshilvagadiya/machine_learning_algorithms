# 📐 The Mathematics of AdaBoost: Sample Weights & Estimator Weight $\alpha$

Here is the exact mathematical formulation of the Discrete AdaBoost algorithm (AdaBoost.M1).

---

## 1. Algorithm Step-by-Step

Let the training dataset be $\{(x_i, y_i)\}_{i=1}^N$ with labels $y_i \in \{-1, +1\}$.

### Step 1: Initialize Uniform Weights
$$w_i^{(1)} = \frac{1}{N}, \quad \forall i \in \{1, \dots, N\}$$

### Step 2: Sequential Loop for $m = 1, 2, \dots, M$
1. **Fit Weak Learner:** Train $h_m(x)$ using weights $w^{(m)}$.
2. **Compute Weighted Error Rate $\epsilon_m$:**
   $$\epsilon_m = \frac{\sum_{i=1}^{N} w_i^{(m)} \cdot \mathbb{I}(y_i \ne h_m(x_i))}{\sum_{i=1}^{N} w_i^{(m)}}$$
   *(Note: If $\epsilon_m \ge 0.5$, stop or invert predictions).*

3. **Compute Estimator Voting Weight $\alpha_m$:**
   $$\alpha_m = \frac{1}{2} \ln \Big( \frac{1 - \epsilon_m}{\epsilon_m} \Big)$$
   - When $\epsilon_m \to 0$ (near-perfect learner), $\alpha_m \to +\infty$.
   - When $\epsilon_m = 0.5$ (pure random guess), $\alpha_m = 0$ (no influence on ensemble).

4. **Update Sample Weights for Next Iteration:**
   $$w_i^{(m+1)} = w_i^{(m)} \exp \big( -\alpha_m y_i h_m(x_i) \big)$$
   - If $y_i = h_m(x_i)$ (correct): $w_i^{(m+1)} = w_i^{(m)} e^{-\alpha_m}$ (decreased).
   - If $y_i \ne h_m(x_i)$ (misclassified): $w_i^{(m+1)} = w_i^{(m)} e^{+\alpha_m}$ (amplified!).

5. **Renormalize Weights:**
   $$w_i^{(m+1)} \leftarrow \frac{w_i^{(m+1)}}{\sum_{k=1}^{N} w_k^{(m+1)}}$$

---

## 2. Final Strong Classifier Decision Rule
The final ensemble prediction is a weighted linear combination of all $M$ weak learners passed through the sign function:

$$H(x) = \text{sign}\Big( \sum_{m=1}^{M} \alpha_m h_m(x) \Big)$$
