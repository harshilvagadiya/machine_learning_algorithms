# ⚡ Sequential Boosting & Weak Learner Intuition in AdaBoost

Adaptive Boosting (AdaBoost; Freund & Schapire, 1995) was the first practical algorithm to convert a committee of "weak learners" (models that perform only slightly better than random guessing) into a single "strong learner" with arbitrarily high accuracy.

---

## 1. The Core Philosophy: Sequential Error Focus

Unlike Bagging (where models are trained in parallel on independent bootstrap samples), Boosting is fundamentally **sequential**:

```
[Dataset with Equal Weights w_i = 1/N]
                  │
                  ▼
         ┌──────────────────┐
         │ Weak Learner h_1 │ ──────> Error ε_1 > 0
         └────────┬─────────┘
                  │
                  ▼
 [Reweight Samples: Misclassified ↑, Correct ↓]
                  │
                  ▼
         ┌──────────────────┐
         │ Weak Learner h_2 │ ──────> Focuses on hard cases missed by h_1
         └────────┬─────────┘
                  │
                  ▼
 [Reweight Samples: Remaining errors ↑]
                  │
                  ▼
         ┌──────────────────┐
         │ Weak Learner h_M │
         └──────────────────┘
```

At each stage $m \in \{1, \dots, M\}$:
1. The weak learner $h_m$ is fitted on the dataset weighted by sample distribution $w^{(m)}$.
2. Its weighted error rate $\epsilon_m$ is calculated.
3. Observations that $h_m$ misclassified have their weights increased (penalized).
4. Observations that $h_m$ classified correctly have their weights decreased.
5. The next learner $h_{m+1}$ is forced to focus its capacity on the hardest, previously misclassified observations!

---

## 2. Why Decision Stumps are the Ideal Weak Learner

In AdaBoost, the default base learner is a **Decision Stump** (a decision tree with `max_depth=1`):
- A decision stump splits on exactly one feature with a single threshold.
- It has high bias and virtually zero variance.
- It is computationally ultra-fast ($< 0.01$ ms).
- By aggregating $M = 100$ stumps with different feature splits and stage weights $\alpha_m$, AdaBoost builds a complex non-linear boundary while remaining resistant to overfitting!
