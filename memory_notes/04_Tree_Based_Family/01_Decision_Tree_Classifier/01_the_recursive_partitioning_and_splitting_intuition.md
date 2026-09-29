# 01. Recursive Binary Partitioning & The Geometric Splitting Intuition

Welcome to the architectural foundation of the **Decision Tree Classifier**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **how decision trees slice feature space, recursive binary splitting mechanics, and the scale-invariance superpower**.

---

## 1. Geometric Intuition: Slicing Space into Orthogonal Hyper-Boxes

Linear models (Logistic Regression, SVM) separate classes using continuous diagonal hyperplanes ($w_1 x_1 + w_2 x_2 + b = 0$).

A **Decision Tree** takes an entirely different geometric approach:
It slices the $D$-dimensional feature space using **axis-aligned orthogonal cuts** (parallel to the coordinate axes), partitioning space into a collection of rectangular hyper-boxes.

```
       FEATURE SPACE PARTITIONING                      EQUIVALENT BINARY TREE
       
       x2 ▲                                                    [ Root Node ]
          │     Region R1          Region R2                   x1 <= 5.0 ?
          │  ┌───────────────┬───────────────────┐             /         \
          │  │               │   (x2 > 7.0)      │          Yes /           \ No
          │  │    Class 0    │     Class 1       │             /             \
      7.0 ┼──┤               ├───────────────────┤      [ Leaf R1 ]       x2 <= 7.0 ?
          │  │               │                   │        Class 0          /       \
          │  │  (x1 <= 5.0)  │     Class 0       │                      Yes /         \ No
          │  │               │   (x2 <= 7.0)     │                         /           \
          └──┴───────────────┴───────────────────┴──► x1             [ Leaf R3 ]   [ Leaf R2 ]
             0              5.0                                       Class 0       Class 1
```

Inside every final terminal box (Leaf Node), the model outputs the **majority class vote** of all training points that landed within that box.

---

## 2. Anatomy of a Decision Tree

A decision tree consists of three structural node types:

1. **Root Node (Level 0):**
   * The top-most node containing $100\%$ of the training samples.
   * Evaluates all features and all possible split thresholds to find the single best question that maximizes class separation.
2. **Internal Decision Nodes:**
   * Intermediate branching points. Each node asks a single binary true/false question:
     $$\text{Condition: } x_j \le \theta$$
   * Samples evaluating to `True` flow to the left child; `False` flow to the right child.
3. **Terminal / Leaf Nodes:**
   * End points with no further splits. Contains the final prediction:
     $$\hat{y} = \arg\max_k \sum_{i \in \text{Leaf}} \mathbb{I}(y_i = k)$$
   * Class probabilities are simply the empirical sample fractions: $P(y = k \mid \mathbf{x}) = \frac{N_k^{\text{leaf}}}{N^{\text{leaf}}}$.

---

## 3. The Greedy CART Algorithm (Top-Down Induction)

Scikit-Learn implements the **CART (Classification and Regression Trees)** algorithm introduced by Breiman et al. (1984).

CART is a **greedy, top-down recursive heuristic**:
1. At the current node, iterate through **every feature** $j \in \{1, \dots, D\}$.
2. For feature $j$, sort all unique observed values and test candidate midpoints $\theta$.
3. For each candidate cut $(j, \theta)$, partition the node's samples into left ($D_L$) and right ($D_R$) child subsets.
4. Compute the resulting **Impurity Reduction (Gain)**:
   $$\Delta I = I(\text{parent}) - \left( \frac{N_L}{N_P} I(D_L) + \frac{N_R}{N_P} I(D_R) \right)$$
5. Choose the pair $(j^*, \theta^*)$ that delivers the **maximum impurity reduction**.
6. Recursively repeat steps 1–5 on $D_L$ and $D_R$ until a stopping criterion (e.g. `max_depth`, `min_samples_split`) is reached.

> **Why "Greedy"?**
> The algorithm picks the best immediate cut at the current step without looking ahead. It cannot backtrack if a suboptimal current split would have enabled a glorious split two levels deeper.

---

## 4. The Scale-Invariance Superpower

In distance-based models (KNN) and gradient-based models (Logistic Regression, Neural Networks), feature scaling (`StandardScaler`, `MinMaxScaler`) is **mandatory**.

In Decision Trees, **FEATURE SCALING IS 100% UNNECESSARY**:

* Suppose feature $x_1$ is `Annual Salary` ranging from $\$20,000$ to $\$200,000$.
* If the optimal split threshold is at $\$75,000$, all samples with $x_1 \le 75,000$ go left.
* If you apply `StandardScaler()`, multiply by $1,000$, or take $\log(x_1)$, the **order of samples remains strictly identical**.
* The threshold simply shifts to $\theta' = z(75000)$ or $\ln(75000)$. The left/right sample routing does not change by a single data point!

```
  RAW SALARY:          20k ─── 45k ─── 70k ─── [75k CUT] ─── 90k ─── 150k
  LOG TRANSFORMED:    9.90 ── 10.71 ── 11.15 ── [11.22 CUT] ── 11.40 ── 11.91
  (Partitions are 100% mathematically invariant under monotonic transformations!)
```

---

## 5. Architectural Tradeoffs

| Advantage | Limitation |
| :--- | :--- |
| **High Interpretability:** Can be converted into human-readable if-else rules. | **High Variance:** A small change in training data can drastically alter the entire tree structure. |
| **Scale Invariant:** Robust to outliers and unscaled features. | **Orthogonal Slicing Trap:** Struggles with smooth diagonal boundaries ($x_1 + x_2 > 5$). |
| **Handles Non-Linearity:** Naturally models non-linear interactions without feature engineering. | **Overfitting Prone:** Unconstrained trees memorize training noise completely (100% train accuracy). |
