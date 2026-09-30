# 🤝 Ensemble Methods Architecture & Memory Notes

Ensemble learning combines multiple distinct hypotheses to form a single consensus model with substantially lower variance and higher predictive reliability than any individual component.

---

## 🏛️ Ensemble Paradigms Overview

```
                      ┌──────────────────────────────────────────────┐
                      │             ENSEMBLE METHODS                 │
                      └──────────────────────┬───────────────────────┘
                                             │
      ┌──────────────────────┬───────────────┴───────────────┬──────────────────────┐
      │                      │                               │                      │
┌─────▼──────────┐    ┌──────▼─────────┐             ┌───────▼────────┐     ┌───────▼────────┐
│  01. VOTING    │    │  02. STACKING  │             │  03. BAGGING   │     │ 04. EXTRA TREES│
│ Hard vs Soft   │    │ Meta-Learner & │             │ Bootstrap &    │     │ Random Cut-Pts │
│ Condorcet Jury │    │ Out-Of-Fold CV │             │ Random Patches │     │ Fast Variance  │
└────────────────┘    └────────────────┘             └────────────────┘     └────────────────┘
```

---

## 📚 Section Breakdown & Navigation

| Module | Core Theoretical Concept | Primary Benefit | Guide Index |
| :--- | :--- | :--- | :--- |
| [**01. Voting**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/01_Voting/README.md) | Majority rule & weighted probability averaging across heterogeneous models. | Simple, parameter-free combining; robust against isolated model failures. | [01_Voting/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/01_Voting/README.md) |
| [**02. Stacking**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/02_Stacking/README.md) | Meta-learning on Out-Of-Fold (OOF) cross-validated predictions. | Optimal weighting of complementary models without data leakage. | [02_Stacking/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/02_Stacking/README.md) |
| [**03. Bagging & Pasting**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/03_Bagging_and_Pasting/README.md) | Bootstrap Aggregation, Random Subspaces, and Random Patches. | Pure variance reduction; free internal Out-Of-Bag (OOB) validation. | [03_Bagging_and_Pasting/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/03_Bagging_and_Pasting/README.md) |
| [**04. Extra Trees**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/04_Extra_Trees/README.md) | Extremely randomized trees using uniform random split cut-points. | Faster training than standard CART/RF; superior boundary smoothing. | [04_Extra_Trees/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/04_Extra_Trees/README.md) |

---

## ⚖️ Practical Decision Matrix: Which Ensemble to Choose?

| Scenario | Recommended Ensemble | Why? |
| :--- | :--- | :--- |
| Have 3-4 diverse models (LR, RF, SVC) and want quick improvement | **Voting (Soft)** | Zero meta-parameters, immediate gain from Condorcet effect. |
| Kaggle / High-stakes competitive predictive accuracy | **Stacking** | Meta-learner learns complex non-linear blend of models. |
| Have a single unstable, high-variance model (e.g. Deep Tree) | **Bagging** | Suppresses variance down to $\rho \sigma^2$ without increasing bias. |
| Huge continuous tabular dataset; Random Forest is too slow | **Extra Trees** | Eliminates $\mathcal{O}(N \log N)$ threshold sorting per node. |
