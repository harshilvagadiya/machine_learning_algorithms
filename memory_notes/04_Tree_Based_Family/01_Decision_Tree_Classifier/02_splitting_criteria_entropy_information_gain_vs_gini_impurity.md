# 02. Splitting Criteria: Gini Impurity vs. Entropy & Information Gain

Welcome to Guide 02 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **the mathematical definitions of impurity, Shannon entropy vs. Gini, and information gain calculations**.

---

## 1. What is "Impurity"?

A node in a decision tree is **"Pure"** if all training samples belonging to that node share the exact same class label (e.g. $100\%$ Spam).
A node is **"Impure"** if it contains an equal mixture of different classes (e.g. $50\%$ Spam and $50\%$ Ham).

The purpose of every split is to **maximize purity** (minimize impurity) in the child nodes.

```
            MAXIMUM IMPURITY                                  PURE LEAF NODE
       (50% Class 0 / 50% Class 1)                             (100% Class 1)
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │  ●   ○   ●   ○   ●   ○  │                     │  ●   ●   ●   ●   ●   ●  │
         │  ○   ●   ○   ●   ○   ●  │                     │  ●   ●   ●   ●   ●   ●  │
         └─────────────────────────┘                     └─────────────────────────┘
            Gini = 0.50 | Entropy = 1.0                     Gini = 0.00 | Entropy = 0.0
```

---

## 2. Gini Impurity (The Scikit-Learn Default)

**Gini Impurity** measures the probability that a randomly chosen element from the node would be incorrectly labeled if it were randomly labeled according to the class distribution in the subset.

$$G = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is the fraction of samples belonging to class $k$ in that node:
$$p_k = \frac{N_k}{N_{\text{node}}}$$

### Binary Classification Example ($K=2$):
* **Pure Node (100% Class 1, 0% Class 0):**
  $$G = 1 - (1.0^2 + 0.0^2) = 1 - 1 = \mathbf{0.0} \quad (\text{Minimum Impurity})$$
* **Maximum Confusion (50% Class 1, 50% Class 0):**
  $$G = 1 - (0.5^2 + 0.5^2) = 1 - (0.25 + 0.25) = \mathbf{0.50} \quad (\text{Maximum Impurity})$$

---

## 3. Shannon Entropy & Information Gain

Derived from Claude Shannon's Information Theory (1948), **Entropy** measures the average amount of "surprise" or "uncertainty" in the node:

$$H = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

*(Note: If $p_k = 0$, by mathematical convention $0 \log_2(0) \equiv 0$)*.

### Information Gain ($IG$):
Information gain is the reduction in entropy achieved by partitioning a node into child subsets:

$$IG(D, j, \theta) = H(D) - \left( \frac{N_L}{N} H(D_L) + \frac{N_R}{N} H(D_R) \right)$$

* **Pure Node (100% Class 1):**
  $$H = -(1.0 \log_2(1.0) + 0) = \mathbf{0.0}$$
* **Maximum Confusion (50/50):**
  $$H = -(0.5 \log_2(0.5) + 0.5 \log_2(0.5)) = -(-0.5 - 0.5) = \mathbf{1.0}$$

---

## 4. Gini vs. Entropy: The Definitive Comparison

```
  IMPURITY CURVES (BINARY CASE p in [0, 1])
  
    Impurity
       ▲
   1.0 ┼                      * * *  Entropy H(p)
   0.8 ┼                  * *       * *
   0.6 ┼                *               *
   0.5 ┼──────────────*───────────────────*────── Gini G(p) Peak at 0.5
   0.4 ┼            *                       *
   0.2 ┼          *                           *
   0.0 ┼─────────*─────────────────────────────*────► p (Probability of Class 1)
       0.0      0.2        0.5        0.8     1.0
```

```
  ┌───────────────────────┬───────────────────────────┬───────────────────────────┐
  │ PROPERTY              │ GINI IMPURITY             │ SHANNON ENTROPY           │
  ├───────────────────────┼───────────────────────────┼───────────────────────────┤
  │ Scikit-Learn Argument │ criterion='gini' (Default)│ criterion='entropy'       │
  │ Maximum Value (Binary)│ 0.50                      │ 1.00                      │
  │ Computation Speed     │ FAST (multiplications)    │ SLOWER (log2 calculations)│
  │ Prone to Extreme Pure │ Slightly favors most      │ Slightly more balanced    │
  │ Splits?               │ frequent class.           │ splits.                   │
  └───────────────────────┴───────────────────────────┴───────────────────────────┘
```

> **Senior Developer Verdict:**
> In $98\%$ of real-world datasets, `criterion='gini'` and `criterion='entropy'` produce **virtually identical tree structures and accuracies**. Gini is standard in production because it computes significantly faster by avoiding logarithmic operations.
