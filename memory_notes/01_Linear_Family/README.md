# 🟢 Machine Learning Master Notes: 01 - Linear / Linear-Family Algorithms

> **The Linear Family Foundation:**
> Machine learning ki duniya ka sabse fundamental aur mathematically solid foundation **Linear Models** hain.
> In sabhi algorithms ka core engine ek hi simple linear equation par chalta hai:
> $$z = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
> Lekin target type ($y$), loss function, aur regularization penalty ke badalne se yeh 7 alag-alag powerful algorithms ban jaate hain!

---

## 🧭 The Linear Family Master Decision Tree

```
                                  LINEAR / LINEAR-FAMILY MODELS
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     CONTINUOUS TARGET (REGRESSION)                                DISCRETE TARGET (CLASSIFICATION)
       "Ghar ki price kitni hai?"                                      "Patient pass hoga ya fail?"
                 │                                                             │
                 ▼                                                             ▼
         DATA SIZE & NOISE?                                             DATASET SCALE?
                 │                                                             │
     ┌───────────┼───────────┐                                         ┌───────┴───────┐
     ▼           ▼           ▼                                         ▼               ▼
NO PENALTY   L2 PENALTY  L1 PENALTY                              SMALL / MEDIUM     BIG DATA (> 1M)
  (OLS)      (Collinear)  (Feature Selection)                   (Standard Solver)  (Streaming/Online)
     │           │           │                                         │               │
     ▼           ▼           ▼                                         ▼               ▼
01_Linear   03_Ridge    04_Lasso                                 02_Logistic     07_SGD_Classifier
Regression  Regression  Regression                               Regression
                 │
                 ▼ (Both L1 + L2)
           05_ElasticNet
                 │
                 ▼ (Big Data / Online Streams)
           06_SGD_Regressor
```

---

## 📊 The Linear-Family Master Suite

| # | Algorithm Folder | Target | Loss Function | Penalty (Regularization) | Best Used For |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **00** | [00_Common_Linear_Family](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/) | **Universal** | **Data Preprocessing & Foundations** | **Zero-Leakage Architecture** | **Common 7-step boilerplate for ALL linear algorithms!** |
| **01** | [01_Linear_Regression](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/01_Linear_Regression/) | Continuous | Mean Squared Error (MSE) | None (Pure OLS) | Baseline regression, strictly linear data with no multicollinearity |
| **02** | [02_Logistic_Regression](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/02_Logistic_Regression/) | Binary / Multiclass | Binary Cross-Entropy (Log-Loss) | $L_2$ (Default) or $L_1$ | High-stakes interpretable classification (Banking, Healthcare, Credit) |
| **03** | [03_Ridge_Regression](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/03_Ridge_Regression/) | Continuous | $\text{MSE} + \alpha \sum w_j^2$ | $L_2$ (Ridge / Weight Decay) | Multicollinearity present, many small-to-moderate feature effects |
| **04** | [04_Lasso_Regression](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/) | Continuous | $\text{MSE} + \alpha \sum \|w_j\|$ | $L_1$ (Lasso / Sparsity) | High-dimensional data ($p \gg n$), automatic feature selection (zeros out weights) |
| **05** | [05_ElasticNet_Regression](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/05_ElasticNet_Regression/) | Continuous | $\text{MSE} + \alpha \cdot \rho L_1 + \frac{\alpha(1-\rho)}{2} L_2$ | $L_1 + L_2$ Combined | Correlated feature groups, genomic data, avoids Lasso's arbitrary feature dropping |
| **06** | [06_SGD_Regressor](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/06_SGD_Regressor/) | Continuous | Squared Loss / Huber / Epsilon | Flexible ($L_1$, $L_2$, ElasticNet) | Massive datasets (10 lakh+ rows), Out-of-Core memory streaming (`partial_fit`) |
| **07** | [07_SGD_Classifier](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/) | Discrete | Log-loss / Hinge (Linear SVM) | Flexible ($L_1$, $L_2$, ElasticNet) | Large-scale classification (Massive text classification, real-time streaming data) |

---

## 🗂️ Algorithms in this Family

0. 📂 **`00_Common_Linear_Family/`** — Universal 7-Step Foundations (Libraries, Ingestion, Segregation, Hygiene, Zero-Leakage Split, Target Profiler, ColumnTransformer).
1. 📂 **`01_Linear_Regression/`** — Standard 8-Step Regression Protocol (Data Loading to Evaluation).
2. 📂 **`02_Logistic_Regression/`** — Complete 13-Step Classification Protocol (Preprocessing to Production Pipeline & Serialization).
3. 📂 **`03_Ridge_Regression/`** — L2 Regularization, Multicollinearity Shrinkage, $\alpha$ tuning via `RidgeCV`.
4. 📂 **`04_Lasso_Regression/`** — L1 Regularization, Sparsity, Feature Selection, $\alpha$ tuning via `LassoCV`.
5. 📂 **`05_ElasticNet_Regression/`** — Combined L1+L2, `l1_ratio` tuning via `ElasticNetCV`.
6. 📂 **`06_SGD_Regressor/`** — Stochastic Gradient Descent for Large-Scale Continuous Regression, Learning Rate Schedules.
7. 📂 **`07_SGD_Classifier/`** — Stochastic Gradient Descent for Large-Scale Classification, Streaming `partial_fit`.

