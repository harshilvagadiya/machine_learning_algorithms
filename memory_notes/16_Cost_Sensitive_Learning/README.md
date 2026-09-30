# ⚖️ Cost-Sensitive Learning & Asymmetric Classification Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Standard classification assumes symmetric 0-1 loss:
$$\mathcal{L}(\hat{y}, y) = \mathbb{I}(\hat{y} \ne y)$$
In real production systems, **false negatives and false positives carry drastically different financial, medical, or legal penalties**:
- A False Negative in cancer screening ($\text{Cost} = \$1,000,000$ + patient morbidity) vs a False Positive ($\text{Cost} = \$200$ for follow-up biopsy).
- A False Negative in financial fraud detection ($\text{Cost} = \$50,000$ stolen) vs a False Positive ($\text{Cost} = \$5$ for customer SMS alert).

**Cost-Sensitive Learning** incorporates an explicit asymmetric cost matrix $\mathbf{C} \in \mathbb{R}^{K \times K}$ directly into the loss function or decision threshold to minimize **Expected Operational Cost**, rather than maximizing symmetric accuracy.

---

## 2. Asymmetric Cost Matrix & Bayes Optimal Decision Rule

$$\mathbf{C} = \begin{pmatrix} C(0, 0) & C(0, 1) \\ C(1, 0) & C(1, 1) \end{pmatrix}$$
where $C(\hat{y}, y)$ is the cost of predicting $\hat{y}$ when true state is $y$.
Typically, correct classifications incur zero cost: $C(0, 0) = C(1, 1) = 0$.
Let $C_{\text{FN}} = C(0, 1)$ (cost of False Negative) and $C_{\text{FP}} = C(1, 0)$ (cost of False Positive).

### The Bayes Optimal Cost-Minimizing Threshold:
Expected cost of predicting class 1 given posterior probability $P(y = 1 \mid \mathbf{x}) = p$:
$$\mathcal{R}(1 \mid \mathbf{x}) = (1 - p) C_{\text{FP}}$$
Expected cost of predicting class 0:
$$\mathcal{R}(0 \mid \mathbf{x}) = p C_{\text{FN}}$$

Predict class 1 if and only if $\mathcal{R}(1 \mid \mathbf{x}) < \mathcal{R}(0 \mid \mathbf{x})$, yielding the optimal decision threshold $\tau^*$:
$$\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}} = \frac{1}{1 + \frac{C_{\text{FN}}}{C_{\text{FP}}}}$$

* When $C_{\text{FN}} = C_{\text{FP}}$ (symmetric loss), $\tau^* = \frac{1}{2} = 0.50$.
* When $C_{\text{FN}} = 10 \cdot C_{\text{FP}}$ (10x penalty for false negatives), $\tau^* = \frac{1}{11} \approx 0.0909$.

---

## 3. Implementation Paradigms

1. **Loss-Function Weighting (Algorithm-Level):**
   Weighting training instances or loss functions directly via class weights:
   $$w_1 = \frac{N}{2 \cdot N_1} \cdot \frac{C_{\text{FN}}}{C_{\text{FP}}}$$
   Implemented via `class_weight='balanced'` or custom class weight dictionaries in Scikit-Learn, LightGBM, and XGBoost.
2. **Post-Hoc Threshold Tuning (Decision-Level):**
   Train a well-calibrated probabilistic model (via Platt Scaling / Isotonic Regression) and apply the optimal threshold $\tau^*$ during production scoring.
3. **Cost-Proportionate Resampling (Data-Level):**
   Re-sample instances in proportion to their misclassification penalty $p_i \propto C(\hat{y}_i, y_i)$.

---

## 4. Production Engineering & Serving Latency
- **Sub-50ms SLA Compliance:** Threshold moving and loss weighting add zero operational compute latency at serving time:
  $$\hat{y} = \mathbb{I}(P(y = 1 \mid \mathbf{x}) \ge \tau^*) \implies \text{Latency} \le 0.12 \text{ ms}$$
- **Evaluation Metrics:** Never evaluate cost-sensitive models with Accuracy. Use:
  $$\text{Total Cost} = \sum_{i=1}^N C(\hat{y}_i, y_i) = \text{FN} \cdot C_{\text{FN}} + \text{FP} \cdot C_{\text{FP}}$$
  $$\text{Expected Value Lift} = \text{Cost}_{\text{baseline}} - \text{Cost}_{\text{model}}$$
