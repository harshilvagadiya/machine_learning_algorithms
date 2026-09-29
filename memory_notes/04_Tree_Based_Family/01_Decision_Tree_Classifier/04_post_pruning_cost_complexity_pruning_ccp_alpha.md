# 04. Post-Pruning: Minimal Cost-Complexity Pruning (`ccp_alpha`)

Welcome to Guide 04 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **post-pruning theory, the cost-complexity objective function, and tuning `ccp_alpha`**.

---

## 1. Pre-Pruning vs. Post-Pruning

* **Pre-Pruning (Early Stopping):** Stops tree growth early (e.g. `max_depth=4`).
  * *Flaw:* The **Horizon Effect**. A split that looks weak today might have unlocked a decisive, pure split in the very next step. Pre-pruning discards it blindly.
* **Post-Pruning (Grow Full, Then Prune):**
  * Grow the tree to its maximum unconstrained depth (memorizing everything).
  * Work backwards from the bottom up, cutting off branches that provide negligible statistical benefit on validation data.

---

## 2. Minimal Cost-Complexity Pruning Theory

Minimal Cost-Complexity Pruning (Breiman et al., 1984) formalizes pruning via a **regularized objective function**:

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

Where:
* $R(T)$: Total empirical training misclassification cost (impurity) of tree $T$.
* $|T|$: Number of terminal leaf nodes in tree $T$ (model complexity).
* $\alpha$ (`ccp_alpha`): The complexity penalty parameter (regularization strength).

```
                 COST-COMPLEXITY OBJECTIVE FUNCTION
                 
            R_α(T)  =    R(T)      +      α  *  |T|
                        ──────            ──────────
                      Fit to Data      Penalty for Size
```

### Behavior across $\alpha$:
* $\alpha = 0$: Standard unconstrained tree (massive tree, $|T|$ is ignored).
* $\alpha > 0$: Branches are trimmed sequentially if their impurity reduction does not justify their leaf penalty.
* $\alpha \to \infty$: Total collapse. The tree is pruned into a single root node (predicting global prior).

---

## 3. Finding the Pruning Path with `cost_complexity_pruning_path`

Scikit-Learn provides an automated method to calculate the exact sequence of effective alphas:

```python
from sklearn.tree import DecisionTreeClassifier

# 1. Fit unconstrained tree
clf = DecisionTreeClassifier(random_state=42)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# 2. ccp_alphas contains the exact mathematical inflection points!
# Last alpha prunes the root node, so exclude it: ccp_alphas[:-1]
```

```python
# 3. Train trees across ccp_alphas to find peak test accuracy
models = []
for a in ccp_alphas[:-1]:
    t = DecisionTreeClassifier(random_state=42, ccp_alpha=a)
    t.fit(X_train, y_train)
    models.append(t)

test_scores = [accuracy_score(y_test, m.predict(X_test)) for m in models]
best_idx = np.argmax(test_scores)
optimal_alpha = ccp_alphas[best_idx]
print(f"Optimal ccp_alpha: {optimal_alpha:.5f}")
```

> **Enterprise Impact:**
> Using `ccp_alpha` yields a tree with **optimal mathematical sparsity**, avoiding arbitrary manual guesswork of depth limits while maximizing test generalization.
