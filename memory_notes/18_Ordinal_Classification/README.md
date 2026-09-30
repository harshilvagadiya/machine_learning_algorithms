# 🥇 Ordinal Classification (Frank & Hall / Proportional Odds) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Ordinal data exhibits a natural, strict mathematical ordering between discrete categories, but the numerical distances between adjacent ranks are undefined:
$$y \in \{0, 1, \dots, K-1\} \quad \text{where } 0 \prec 1 \prec \dots \prec K-1$$
- Medical Staging: Cancer Stage 0 < Stage I < Stage II < Stage III < Stage IV.
- Sommelier Ratings: Poor (3) < Fair (4) < Average (5) < Good (6) < Premium (7) < Exceptional (8).
- Customer Surveys: Strongly Disagree (1) < Disagree (2) < Neutral (3) < Agree (4) < Strongly Agree (5).

Standard Multi-Class Softmax penalizes errors symmetrically: confusing Stage 0 with Stage IV incurs the identical cross-entropy loss as confusing Stage 0 with Stage I.
Conversely, standard regression imposes an arbitrary Euclidean metric, assuming the gap between Fair and Good equals the gap between Premium and Exceptional.

---

## 2. Frank & Hall (2001) Threshold Decomposition
Frank & Hall converts a $K$-class ordinal classification task into $K-1$ sequential binary classification models $C_0, C_1, \dots, C_{K-2}$.
Classifier $C_k$ is trained on the binary indicator:
$$\tilde{y}_k = \mathbb{I}(y > k)$$

### Conditional Probability Reconstruction:
Let $P_k = P(y > k \mid \mathbf{x}) = C_k(\mathbf{x})$. The monotonic class probabilities are recovered via:
$$P(y = 0 \mid \mathbf{x}) = 1 - P_0$$
$$P(y = k \mid \mathbf{x}) = P_{k-1} - P_k \quad \forall 1 \le k < K-1$$
$$P(y = K-1 \mid \mathbf{x}) = P_{K-2}$$

Non-monotonic probabilities resulting from finite sample variance are clipped ($P \ge 0$) and normalized so $\sum_{k=0}^{K-1} P(y = k \mid \mathbf{x}) = 1$.
Final discrete rank prediction:
$$\hat{y} = \arg\max_{k \in \{0, \dots, K-1\}} P(y = k \mid \mathbf{x})$$

---

## 3. Evaluation Metrics for Ordinal Models

1. **Quadratic Weighted Kappa (QWK):**
   Measures inter-rater agreement penalized quadratically by distance:
   $$\kappa = 1 - \frac{\sum_{i, j} w_{ij} O_{ij}}{\sum_{i, j} w_{ij} E_{ij}}, \quad w_{ij} = \frac{(i - j)^2}{(K - 1)^2}$$
2. **Mean Absolute Error of Ranks (MAE-Rank):**
   $$\text{MAE}_{\text{rank}} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
3. **Off-by-1 Accuracy:** Percentage of predictions where $|\hat{y} - y| \le 1$.

---

## 4. Production Engineering & Serving Latency
- **Model Size:** Consists of exactly $K-1$ lightweight binary linear or tree estimators. Total footprint is under 20 KB.
- **Serving Latency SLA:**
  $$\text{Mean Serving Latency} \le 0.45 \text{ ms (P99 < 1.10 ms)}$$
- **Monotonicity Guarantees:** By decomposing into cumulative exceedance probabilities, the model strictly penalizes large rank inversions, eliminating catastrophic edge misclassifications.
