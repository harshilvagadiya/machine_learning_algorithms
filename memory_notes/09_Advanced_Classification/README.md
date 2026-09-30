# 🎯 Advanced Classification Strategies & Calibration — Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Real-world classification frequently breaks standard binary, well-calibrated assumptions:
1. **Multi-Class Complexity ($K > 2$):** Algorithms designed natively for binary separation (like SVM, Perceptron, AdaBoost.M1) require systematic multi-class meta-strategies.
2. **Confidence Miscalibration:** Deep models, Naive Bayes, and SVMs output decision scores or overconfident softmax probabilities that do **not** reflect true posterior probability $P(Y=1 \mid \hat{P}=p) \neq p$.

---

## 2. Multi-Class Meta-Strategies

| Strategy | Decomposition Mechanism | Number of Classifiers | Decision Rule | Time Complexity | Best Fit Algorithms |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **One-vs-Rest (OvR / One-vs-All)** | Train $K$ independent classifiers, each separating class $k$ from all remaining $K-1$ classes | $K$ | $\hat{y} = \arg\max_k f_k(\mathbf{x})$ | $\mathcal{O}(K \cdot \text{Cost}(N))$ | Logistic Regression, SGDClassifier |
| **One-vs-One (OvO)** | Train a binary classifier for every distinct pair of classes $(j, k)$ | $\frac{K(K-1)}{2}$ | Majority vote tally across all pairwise duels | $\mathcal{O}(K^2 \cdot \text{Cost}(N/K))$ | Support Vector Machines (SVC with non-linear kernels) |
| **Error-Correcting Output Codes (ECOC)** | Encodes classes into binary codewords of length $L > K$ | $L$ | Closest codeword via Hamming or Euclidean distance | $\mathcal{O}(L \cdot \text{Cost}(N))$ | Resilient to individual classifier errors |

---

## 3. Probability Calibration

### Why Calibration Matters:
If a medical diagnostic or credit risk model predicts a fraud risk of $0.90$, exactly $90\%$ of samples receiving that score should be positive. If only $50\%$ are positive, downstream decision thresholds and expected-value calculations produce catastrophic business errors.

### Calibration Techniques:

1. **Platt Scaling (Sigmoid Method - Platt, 1999):**
   - Fits a scalar logistic regression model on the model's raw margins $f(\mathbf{x})$ using an independent validation split:
     $$P(Y = 1 \mid f(\mathbf{x})) = \frac{1}{1 + \exp(A \cdot f(\mathbf{x}) + B)}$$
   - **Parameters:** Only 2 parameters $(A, B)$ estimated via Maximum Likelihood.
   - **Best For:** Small validation sets ($N < 1000$), models with sigmoidal distortion (e.g. SVM margin).

2. **Isotonic Regression (Non-Parametric - Zadrozny & Elkan, 2002):**
   - Fits a non-decreasing, piece-wise constant monotonic step function minimizing squared loss:
     $$\min \sum_{i=1}^m (y_i - m(f(\mathbf{x}_i)))^2 \quad \text{subject to} \quad m(f_i) \le m(f_j) \text{ whenever } f_i \le f_j$$
   - Solved in $\mathcal{O}(N)$ using the **Pool Adjacent Violators (PAV)** algorithm.
   - **Best For:** Large validation datasets ($N \ge 1000$), arbitrary monotonic calibration distortions.

---

## 4. Evaluation: Expected Calibration Error (ECE) & Reliability Diagrams
Group predictions into $M$ equal-width probability bins $B_m \subset (0, 1]$:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|$$
- **Reliability Diagrams:** Plots observed sample accuracy $\operatorname{acc}(B_m)$ against mean confidence $\operatorname{conf}(B_m)$ with a $45^\circ$ reference diagonal.
- Perfectly calibrated model: $\text{ECE} = 0.0$.
