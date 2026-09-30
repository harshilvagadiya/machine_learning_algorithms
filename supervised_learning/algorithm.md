# 🤖 Supervised Learning — Comprehensive Progress Tracker & Master Directory

A complete, enterprise-grade roadmap and index of all **Supervised Machine Learning Algorithm Families** ($f: \mathcal{X} \to \mathcal{Y}$) and specialized architectures.

Every family folder in `supervised_learning/` is prefixed with an intuitive **sequential learning step number (`01_` to `23_`)** matching the corresponding detailed theoretical notes in `memory_notes/`.

Every algorithm implementation strictly adheres to the **Enterprise 13-Step Standard**:
1. **4 Real-World Datasets** per algorithm / sub-family (Zero synthetic data).
2. **End-to-End Executed Jupyter Notebooks** with full outputs, convergence logs, and visualization plots.
3. **Serialized `.joblib` Production Pipelines** equipped with zero-leakage `ColumnTransformer` preprocessing.
4. **Sub-50ms Serving Latency Verification** (`assert avg_latency < 50.0 ms`) for real-time inference.
5. **Comprehensive Mathematical Memory Notes** with full KaTeX LaTeX formulations and architectural comparisons.

---

## 🧭 Step-by-Step Learning Path & Directory Index

| Step # | Family Directory (Code & Models) | Memory Notes (Deep Theory) | Core Concepts / Algorithms | Notebooks | Models | Status |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **01** | [`01_Linear_Family/`](file:///home/python/03_Custom_addons/ML/supervised_learning/01_Linear_Family) | [`01_Linear_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family) | OLS Linear, Logistic, Ridge, Lasso, ElasticNet, SGD | 26 | 17 | ✅ Done |
| **02** | [`02_Distance_Based/`](file:///home/python/03_Custom_addons/ML/supervised_learning/02_Distance_Based) | [`02_Distance_Based_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/02_Distance_Based_Family) | KNN Classifier, KNN Regressor, KD-Tree, Ball-Tree | 8 | 8 | ✅ Done |
| **03** | [`03_Probabilistic/`](file:///home/python/03_Custom_addons/ML/supervised_learning/03_Probabilistic) | [`03_Probabilistic_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/03_Probabilistic_Family) | Gaussian, Multinomial, Bernoulli Naive Bayes | 12 | 12 | ✅ Done |
| **04** | [`04_Tree_Based/`](file:///home/python/03_Custom_addons/ML/supervised_learning/04_Tree_Based) | [`04_Tree_Based_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/04_Tree_Based_Family) | Decision Trees & Random Forests (Clf & Reg) | 16 | 16 | ✅ Done |
| **05** | [`05_SVM_Family/`](file:///home/python/03_Custom_addons/ML/supervised_learning/05_SVM_Family) | [`05_SVM_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/05_SVM_Family) | Support Vector Classifier (SVC), Support Vector Regressor (SVR) | 8 | 8 | ✅ Done |
| **06** | [`06_Boosting_Family/`](file:///home/python/03_Custom_addons/ML/supervised_learning/06_Boosting_Family) | [`06_Boosting_Family`](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family) | AdaBoost, GradientBoosting, HistGBM, XGBoost, LightGBM, CatBoost | 48 | 48 | ✅ Done |
| **07** | [`07_Ensemble_Methods/`](file:///home/python/03_Custom_addons/ML/supervised_learning/07_Ensemble_Methods) | [`07_Ensemble_Methods`](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods) | Bagging, Extra Trees, Voting, Stacking, Blending | 40 | 40 | ✅ Done |
| **08** | [`08_Advanced_Regression/`](file:///home/python/03_Custom_addons/ML/supervised_learning/08_Advanced_Regression) | [`08_Advanced_Regression`](file:///home/python/03_Custom_addons/ML/memory_notes/08_Advanced_Regression) | Poisson, Gamma, Tweedie, NegBinomial, Robust, Quantile, Poly, GPR | 24 | 24 | ✅ Done |
| **09** | [`09_Advanced_Classification/`](file:///home/python/03_Custom_addons/ML/supervised_learning/09_Advanced_Classification) | [`09_Advanced_Classification`](file:///home/python/03_Custom_addons/ML/memory_notes/09_Advanced_Classification) | One-vs-Rest, One-vs-One, Platt Calibration, Isotonic Calibration | 8 | 8 | ✅ Done |
| **10** | [`10_Neural_Networks/`](file:///home/python/03_Custom_addons/ML/supervised_learning/10_Neural_Networks) | [`10_Neural_Networks`](file:///home/python/03_Custom_addons/ML/memory_notes/10_Neural_Networks) | Perceptron, Multi-Layer Perceptron (MLPClassifier, MLPRegressor) | 12 | 12 | ✅ Done |
| **11** | [`11_Discriminant_Analysis/`](file:///home/python/03_Custom_addons/ML/supervised_learning/11_Discriminant_Analysis) | [`11_Discriminant_Analysis`](file:///home/python/03_Custom_addons/ML/memory_notes/11_Discriminant_Analysis) | Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) | 8 | 8 | ✅ Done |
| **12** | [`12_Bayesian_Regression/`](file:///home/python/03_Custom_addons/ML/supervised_learning/12_Bayesian_Regression) | [`12_Bayesian_Regression`](file:///home/python/03_Custom_addons/ML/memory_notes/12_Bayesian_Regression) | Bayesian Ridge Regression, Automatic Relevance Determination (ARD) | 8 | 8 | ✅ Done |
| **13** | [`13_Nearest_Centroid/`](file:///home/python/03_Custom_addons/ML/supervised_learning/13_Nearest_Centroid) | [`13_Nearest_Centroid`](file:///home/python/03_Custom_addons/ML/memory_notes/13_Nearest_Centroid) | Nearest Centroid Classifier (Rocchio, Shrinkage Thresholds) | 4 | 4 | ✅ Done |
| **14** | [`14_Partial_Least_Squares/`](file:///home/python/03_Custom_addons/ML/supervised_learning/14_Partial_Least_Squares) | [`14_Partial_Least_Squares`](file:///home/python/03_Custom_addons/ML/memory_notes/14_Partial_Least_Squares) | PLS Regression (`PLSRegression`), PLS Canonical (`PLSCanonical`) | 8 | 8 | ✅ Done |
| **15** | [`15_Passive_Aggressive/`](file:///home/python/03_Custom_addons/ML/supervised_learning/15_Passive_Aggressive) | [`15_Passive_Aggressive`](file:///home/python/03_Custom_addons/ML/memory_notes/15_Passive_Aggressive) | Passive-Aggressive Classifier & Regressor (Online Streaming Margin) | 8 | 8 | ✅ Done |
| **16** | [`16_Cost_Sensitive/`](file:///home/python/03_Custom_addons/ML/supervised_learning/16_Cost_Sensitive) | [`16_Cost_Sensitive_Learning`](file:///home/python/03_Custom_addons/ML/memory_notes/16_Cost_Sensitive_Learning) | Cost-Sensitive Classifier (Asymmetric Loss Matrices, Optimal $\tau^*$) | 4 | 4 | ✅ Done |
| **17** | [`17_Multi_Output/`](file:///home/python/03_Custom_addons/ML/supervised_learning/17_Multi_Output) | [`17_Multi_Output`](file:///home/python/03_Custom_addons/ML/memory_notes/17_Multi_Output) | MultiOutputClassifier, ClassifierChain, MultiOutputRegressor, RegressorChain | 16 | 16 | ✅ Done |
| **18** | [`18_Ordinal_Classification/`](file:///home/python/03_Custom_addons/ML/supervised_learning/18_Ordinal_Classification) | [`18_Ordinal_Classification`](file:///home/python/03_Custom_addons/ML/memory_notes/18_Ordinal_Classification) | Frank-Hall Binary Decomposition, Quadratic Weighted Kappa | 4 | 4 | ✅ Done |
| **19** | [`19_Orthogonal_Distance_Regression/`](file:///home/python/03_Custom_addons/ML/supervised_learning/19_Orthogonal_Distance_Regression) | [`19_Orthogonal_Distance_Regression`](file:///home/python/03_Custom_addons/ML/memory_notes/19_Orthogonal_Distance_Regression) | ODR (Total Least Squares for Errors-in-Variables) | 4 | 4 | ✅ Done |
| **20** | [`20_Survival_Analysis/`](file:///home/python/03_Custom_addons/ML/supervised_learning/20_Survival_Analysis) | [`20_Survival_Analysis`](file:///home/python/03_Custom_addons/ML/memory_notes/20_Survival_Analysis) | Cox Proportional Hazards (`CoxPHFitter`), Random Survival Forest (`RSF`) | 8 | 8 | ✅ Done |
| **21** | [`21_Ranking_LTR/`](file:///home/python/03_Custom_addons/ML/supervised_learning/21_Ranking_LTR) | [`21_Ranking_LTR`](file:///home/python/03_Custom_addons/ML/memory_notes/21_Ranking_LTR) | Pointwise, Pairwise RankNet, Listwise LambdaMART (`LGBMRanker`) | 4 | 4 | ✅ Done |
| **22** | [`22_Factorization_Machines/`](file:///home/python/03_Custom_addons/ML/supervised_learning/22_Factorization_Machines) | [`22_Factorization_Machines`](file:///home/python/03_Custom_addons/ML/memory_notes/22_Factorization_Machines) | Factorization Machines (FM $\mathcal{O}(k \cdot p)$), Field-Aware FM (FFM) | 8 | 8 | ✅ Done |
| **23** | [`23_Gaussian_Processes/`](file:///home/python/03_Custom_addons/ML/supervised_learning/23_Gaussian_Processes) | [`08_Advanced_Regression/Gaussian_Process_Regression.md`](file:///home/python/03_Custom_addons/ML/memory_notes/08_Advanced_Regression/Gaussian_Process_Regression.md) | Gaussian Process Classification (GPC - Laplace Approximation) | 4 | 4 | ✅ Done |
| **TOTAL** | **23 Sequenced Learning Steps** | **Full 23-Topic Theory Hub** | **73 Production Sub-Suites Across 23 Algorithm Families** | **290** | **281** | **100% COMPLETE** |

---

## 🏛️ Architectural Principle: 2-Tier Separation

We enforce strict separation between **Supervised Learning Algorithm Families** and **Supporting ML Concepts**:

```text
                               🤖 SUPERVISED LEARNING ($f: \mathcal{X} \to \mathcal{Y}$)
                                                       │
        ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
        │                                              │                                              │
        ▼                                              ▼                                              ▼
  STEPS 01, 08, 12, 19                          STEPS 04, 06, 07                               STEPS 02, 03, 05, 09-23
  LINEAR, GLM & CONTINUOUS                      TREES & ENSEMBLES                              SPECIALIZED PARADIGMS
  ├─ 01_Linear_Family                           ├─ 04_Tree_Based                               ├─ 02_Distance_Based
  ├─ 08_Advanced_Regression                     ├─ 06_Boosting_Family                          ├─ 03_Probabilistic
  ├─ 12_Bayesian_Regression                     └─ 07_Ensemble_Methods                         ├─ 05_SVM_Family
  └─ 19_Orthogonal_Distance_Regression                                                         ├─ 09_Advanced_Classification
                                                                                               ├─ 10_Neural_Networks
                                                                                               ├─ 11_Discriminant_Analysis
                                                                                               ├─ 13_Nearest_Centroid
                                                                                               ├─ 14_Partial_Least_Squares
                                                                                               ├─ 15_Passive_Aggressive
                                                                                               ├─ 16_Cost_Sensitive
                                                                                               ├─ 17_Multi_Output
                                                                                               ├─ 18_Ordinal_Classification
                                                                                               ├─ 20_Survival_Analysis
                                                                                               ├─ 21_Ranking_LTR
                                                                                               ├─ 22_Factorization_Machines
                                                                                               └─ 23_Gaussian_Processes
```

### Supporting ML Concepts (Cross-Cutting Pipelines, NOT Separate Model Families):
- **Cross-Validation Strategies:** K-Fold, Stratified K-Fold, TimeSeriesSplit, GroupKFold.
- **Hyperparameter Optimization:** Grid Search, Randomized Search, Bayesian Search (Optuna).
- **Feature Engineering & Preprocessing:** Imputation, One-Hot / Target Encoding, PowerTransform, Scaling.
- **Model Explainability & Diagnostics:** SHAP values, Permutation Importance, Partial Dependence Plots (PDP).
- **Imbalanced Learning:** SMOTE, Class Weighting, Focal Loss, Threshold Moving.
