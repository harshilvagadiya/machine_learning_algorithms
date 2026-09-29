# 05. Feature Importance: Mean Decrease in Impurity (MDI) vs. Permutation Importance

Welcome to Guide 05 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **how feature importance is computed, the high-cardinality bias trap, and modern permutation importance**.

---

## 1. Mean Decrease in Impurity (MDI / Gini Importance)

When you access `clf.feature_importances_`, Scikit-Learn computes the **Mean Decrease in Impurity (MDI)**:

The importance of feature $j$ is the sum of all impurity reductions brought by feature $j$, weighted by the fraction of samples reaching each splitting node, normalized across all features to sum to $1.0$:

$$\text{Imp}(j) = \sum_{t \in \text{Nodes splitting on } j} \frac{N_t}{N_{\text{total}}} \Delta I(t)$$

Where $\Delta I(t) = I(t) - \left( \frac{N_L}{N_t} I(L) + \frac{N_R}{N_t} I(R) \right)$.

---

## 2. The Dangerous Flaw: High-Cardinality Bias Trap

While MDI is fast to compute, it suffers from a fatal statistical bias:

> [!WARNING]
> **MDI heavily inflates the importance of continuous or high-cardinality categorical features**, even if they contain zero predictive signal!

### Why?
* A binary feature (e.g. `Gender` $\in \{0, 1\}$) offers only **$1$ possible split threshold**.
* A high-cardinality continuous feature (e.g. `Transaction ID` or random noise float) offers **$N-1$ candidate split thresholds**.
* By pure random chance, a feature with thousands of candidate cuts will find spurious splits that accidentally reduce training impurity. MDI will rank pure random noise as the most important feature!

---

## 3. The Enterprise Fix: Permutation Feature Importance

**Permutation Feature Importance** (Breiman, 2001) evaluates feature importance on held-out **test data**:

```
  1. Record baseline test score: Score_base = 0.92
  2. For feature j:
     a. Randomly shuffle (permute) the values of column j in X_test,
        breaking its relationship with the target y_test.
     b. Re-evaluate test score on corrupted test set: Score_perm = 0.74
     c. Importance(j) = Score_base - Score_perm = 0.18 (Drop of 18% points!)
  3. If shuffling column j does not degrade accuracy, feature j is useless!
```

```python
from sklearn.inspection import permutation_importance

perm_imp = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, scoring='accuracy'
)

# Mean importance and standard deviation across repeats
sorted_importances_idx = perm_imp.importances_mean.argsort()[::-1]
```

### MDI vs. Permutation Comparison
| Feature Importance Type | Evaluated On | Susceptible to Cardinality Bias? | Speed |
| :--- | :--- | :--- | :--- |
| **MDI (`feature_importances_`)** | Training Data | **YES** (Severe bias toward float/ID columns) | Instant ($O(1)$) |
| **Permutation Importance** | Test Data | **NO** (Strictly measures true generalization drop) | Moderate ($O(N_{\text{perm}} \cdot D)$) |
