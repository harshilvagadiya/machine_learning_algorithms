# 🧠 Machine Learning Algorithms — Master Architecture & Memory Notes Hub

Welcome to the centralized, production-grade **Machine Learning Memory Notes Knowledge Base**.

Yahan Machine Learning algorithms ko natural mathematical families mein organize kiya gaya hai. Sabhi algorithms ke liye **Universal Pre-Flight Protocol (Data Ingestion to Preprocessing)** ek hi central jagah par define hai, taaki har algorithm mein redundant steps repeat na hon.

---

## 🗺️ Machine Learning Algorithm Families Map

```
                                  SUPERVISED LEARNING
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
00. UNIVERSAL WORKFLOW            01. LINEAR / LINEAR-FAMILY        02. DISTANCE-BASED FAMILY
   (Common to ALL Algorithms)        (Global Hyperplane w^T x + b)     (Local Spatial Proximity)
   ├── 01_Universal Libraries        ├── 01_Linear_Regression          ├── 01_KNN_Classifier
   ├── 02_Data Ingestion & Audit     ├── 02_Logistic_Regression        └── 02_KNN_Regressor
   ├── 02b_String Sanitization       ├── 03_Ridge_Regression
   ├── 03_Feature Segregation        ├── 04_Lasso_Regression
   ├── 04_Missing Values & Hygiene   ├── 05_ElasticNet_Regression
   ├── 05_Zero-Leakage Split         ├── 06_SGD_Regressor
   ├── 06_Target Profiling           └── 07_SGD_Classifier
   └── 07_ColumnTransformer Preproc
```

---

## 📚 Master Directory Navigation

### 🌐 [00. Universal ML Workflow (Common Pre-Flight Checklist)](00_Common_ML_Workflow/README.md)
* **Yeh steps har algorithm mein 100% COMMON hain:**
  1. [Step 1: Universal Libraries Import Guide](00_Common_ML_Workflow/01_universal_libraries_import_guide.md)
  2. [Step 2: Data Ingestion & 3-Sec Audit Guide](00_Common_ML_Workflow/02_data_ingestion_and_3sec_audit_guide.md)
  3. [Step 2b: Data Sanitization & String Cleaning Guide](00_Common_ML_Workflow/02b_data_sanitization_and_string_cleaning_guide.md)
  4. [Step 3: Feature Segregation & Cardinality Guide](00_Common_ML_Workflow/03_feature_segregation_and_cardinality_guide.md)
  5. [Step 4: Missing Values & Duplicates Hygiene Guide](00_Common_ML_Workflow/04_missing_values_and_duplicates_hygiene_guide.md)
  6. [Step 5: Train-Test Split & Zero-Leakage Protocol Guide](00_Common_ML_Workflow/05_train_test_split_and_zero_leakage_guide.md)
  7. [Step 6: Target Analysis & Transformation Guide](00_Common_ML_Workflow/06_target_analysis_and_transformation_guide.md)
  8. [Step 7: Master ColumnTransformer Preprocessing Guide](00_Common_ML_Workflow/07_master_columntransformer_preprocessing_guide.md)

---

### 🟢 [01. Linear Family Hub](01_Linear_Family/README.md)
* **[01_Linear_Regression](01_Linear_Family/01_Linear_Regression/README.md):** Pure OLS, Normal Equation, MSE optimization.
* **[02_Logistic_Regression](01_Linear_Family/02_Logistic_Regression/README.md):** Sigmoid / Logit, Log-Loss, odds ratios, threshold tuning.
* **[03_Ridge_Regression](01_Linear_Family/03_Ridge_Regression/README.md):** $L_2$ Regularization, weight decay, solving multicollinearity.
* **[04_Lasso_Regression](01_Linear_Family/04_Lasso_Regression/README.md):** $L_1$ Regularization, automatic feature selection (sparsity).
* **[05_ElasticNet_Regression](01_Linear_Family/05_ElasticNet_Regression/README.md):** Combined $L_1 + L_2$ penalties, handling correlated feature clusters.
* **[06_SGD_Regressor](01_Linear_Family/06_SGD_Regressor/README.md):** Stochastic Gradient Descent for large datasets, streaming `partial_fit`.
* **[07_SGD_Classifier](01_Linear_Family/07_SGD_Classifier/README.md):** Chameleon classifier (Log-loss, Hinge, Perceptron), out-of-core online classification.

---

### 🟡 [02. Distance-Based Family Hub](02_Distance_Based_Family/README.md)
* **[01_KNN_Classifier](02_Distance_Based_Family/01_KNN_Classifier/README.md):**
  * Guide 01: [The Lazy Learner & Non-Parametric Intuition](02_Distance_Based_Family/01_KNN_Classifier/01_the_lazy_learner_and_non_parametric_intuition.md)
  * Guide 02: [Distance Metrics & Geometric Math](02_Distance_Based_Family/01_KNN_Classifier/02_distance_metrics_and_geometric_math.md)
  * Guide 03: [Bias-Variance Tradeoff of K & Neighbor Weighting](02_Distance_Based_Family/01_KNN_Classifier/03_the_bias_variance_tradeoff_of_k_and_weighting.md)
  * Guide 04: [The Curse of Dimensionality & Mitigation](02_Distance_Based_Family/01_KNN_Classifier/04_curse_of_dimensionality_and_mitigation.md)
  * Guide 05: [Search Algorithms — Brute vs KD-Tree vs Ball-Tree](02_Distance_Based_Family/01_KNN_Classifier/05_search_algorithms_brute_vs_kdtree_vs_balltree.md)
  * Guide 06: [Hyperparameter Tuning & Non-Linear Decision Boundaries](02_Distance_Based_Family/01_KNN_Classifier/06_hyperparameter_tuning_and_decision_boundaries.md)
  * Guide 07: [Production Pipeline, Latency SLAs & Modern ANN Migration](02_Distance_Based_Family/01_KNN_Classifier/07_production_pipeline_latency_and_ann.md)
* **[02_KNN_Regressor](02_Distance_Based_Family/02_KNN_Regressor/README.md):**
  * Guide 01: [The Local Mean & Non-Parametric Regression Intuition](02_Distance_Based_Family/02_KNN_Regressor/01_the_local_mean_and_interpolation_intuition.md)
  * Guide 02: [Uniform vs Distance-Weighted Surfaces](02_Distance_Based_Family/02_KNN_Regressor/02_uniform_vs_distance_weighted_surfaces.md)
  * Guide 03: [The Extrapolation Disaster & Critical Failure Modes](02_Distance_Based_Family/02_KNN_Regressor/03_the_extrapolation_disaster_and_failure_modes.md)
  * Guide 04: [The Bias-Variance Tradeoff of K in Regression](02_Distance_Based_Family/02_KNN_Regressor/04_bias_variance_tradeoff_of_k_in_regression.md)
  * Guide 05: [Regression Evaluation Metrics & Residual Diagnostics](02_Distance_Based_Family/02_KNN_Regressor/05_regression_evaluation_metrics_and_residuals.md)
  * Guide 06: [Hyperparameter Tuning & Spatial Tree Search](02_Distance_Based_Family/02_KNN_Regressor/06_hyperparameter_tuning_and_tree_search_optimization.md)
  * Guide 07: [Production Pipeline, Serialization & Live Inference](02_Distance_Based_Family/02_KNN_Regressor/07_production_pipeline_and_live_inference.md)

---

### 🔵 [03. Probabilistic Family Hub](03_Probabilistic_Family/README.md)
* **[01_Naive_Bayes](03_Probabilistic_Family/01_Naive_Bayes/README.md):**
  * Guide 01: [Bayes' Theorem: Prior, Likelihood, & Posterior](03_Probabilistic_Family/01_Naive_Bayes/01_bayes_theorem_prior_likelihood_posterior.md)
  * Guide 02: [The "Naive" Conditional Independence Assumption](03_Probabilistic_Family/01_Naive_Bayes/02_the_naive_conditional_independence_assumption.md)
  * Guide 03: [The Zero-Frequency Catastrophe & Laplace Smoothing](03_Probabilistic_Family/01_Naive_Bayes/03_the_zero_frequency_problem_and_laplace_smoothing.md)
  * Guide 04: [Gaussian Naive Bayes & Bell Curve Distributions](03_Probabilistic_Family/01_Naive_Bayes/04_gaussian_nb_continuous_features_and_normal_distribution.md)
  * Guide 05: [Multinomial Naive Bayes, Count Vectors, & Text NLP](03_Probabilistic_Family/01_Naive_Bayes/05_multinomial_nb_count_vectors_and_text_classification.md)
  * Guide 06: [Bernoulli & Complement Naive Bayes for Imbalanced Data](03_Probabilistic_Family/01_Naive_Bayes/06_bernoulli_and_complement_nb_for_imbalanced_data.md)
  * Guide 07: [Production Pipelines, Streaming partial_fit, & Sub-Millisecond SLAs](03_Probabilistic_Family/01_Naive_Bayes/07_production_pipeline_vectorization_and_live_inference.md)

---

### 🟣 [04. Tree-Based Family Hub](04_Tree_Based_Family/README.md)
* **[01_Decision_Tree_Classifier](04_Tree_Based_Family/01_Decision_Tree_Classifier/README.md):**
  * Guide 01: [Recursive Binary Partitioning & Slicing](04_Tree_Based_Family/01_Decision_Tree_Classifier/01_the_recursive_partitioning_and_splitting_intuition.md)
  * Guide 02: [Splitting Criteria: Gini vs Entropy](04_Tree_Based_Family/01_Decision_Tree_Classifier/02_splitting_criteria_entropy_information_gain_vs_gini_impurity.md)
  * Guide 03: [The Overfitting Monster & Pre-Pruning](04_Tree_Based_Family/01_Decision_Tree_Classifier/03_the_overfitting_monster_and_pre_pruning_hyperparameters.md)
  * Guide 04: [Minimal Cost-Complexity Post-Pruning (ccp_alpha)](04_Tree_Based_Family/01_Decision_Tree_Classifier/04_post_pruning_cost_complexity_pruning_ccp_alpha.md)
  * Guide 05: [Feature Importance: MDI vs Permutation Importance](04_Tree_Based_Family/01_Decision_Tree_Classifier/05_feature_importance_impurity_decrease_vs_permutation.md)
  * Guide 06: [Tree Visualization & Decision Surfaces](04_Tree_Based_Family/01_Decision_Tree_Classifier/06_tree_visualization_and_decision_surfaces.md)
  * Guide 07: [Production Pipelines, Latency SLAs (< 0.1ms), & SQL Export](04_Tree_Based_Family/01_Decision_Tree_Classifier/07_production_pipeline_tree_latency_and_deployment.md)

---

### 🛡️ [05. Support Vector Machine (SVM) Family Hub](05_SVM_Family/README.md)
* **[01_Support_Vector_Classifier](05_SVM_Family/01_Support_Vector_Classifier/README.md):**
  * Guide 01: [Geometric Margin & Hyperplane Intuition](05_SVM_Family/01_Support_Vector_Classifier/01_geometric_margin_and_hyperplane_intuition.md)
  * Guide 02: [The Primal & Dual Optimization Problem](05_SVM_Family/01_Support_Vector_Classifier/02_the_primal_and_dual_optimization_problem.md)
  * Guide 03: [The Kernel Trick (RBF, Poly, Sigmoid)](05_SVM_Family/01_Support_Vector_Classifier/03_the_kernel_trick_rbf_poly_sigmoid.md)
  * Guide 04: [Hyperparameter Dynamics: C and Gamma](05_SVM_Family/01_Support_Vector_Classifier/04_hyperparameter_dynamics_c_and_gamma.md)
  * Guide 05: [The Critical StandardScaler Rule](05_SVM_Family/01_Support_Vector_Classifier/05_the_critical_standard_scaler_rule.md)
  * Guide 06: [Computational Complexity & Scaling Limits](05_SVM_Family/01_Support_Vector_Classifier/06_computational_complexity_and_scaling_limits.md)
  * Guide 07: [Production Pipeline & Latency Benchmarks](05_SVM_Family/01_Support_Vector_Classifier/07_production_pipeline_and_latency_benchmarks.md)
* **[02_Support_Vector_Regressor](05_SVM_Family/02_Support_Vector_Regressor/README.md):**
  * Guide 01: [The Epsilon-Insensitive Tube Intuition](05_SVM_Family/02_Support_Vector_Regressor/01_the_epsilon_insensitive_tube_intuition.md)
  * Guide 02: [Mathematical Formulation & Slack Variables](05_SVM_Family/02_Support_Vector_Regressor/02_mathematical_formulation_and_slack_variables.md)
  * Guide 03: [Kernel SVR & Non-Linear Function Approximation](05_SVM_Family/02_Support_Vector_Regressor/03_kernel_svr_and_non_linear_function_approximation.md)
  * Guide 04: [The Hyperparameter Trio (C, Epsilon, Gamma)](05_SVM_Family/02_Support_Vector_Regressor/04_hyperparameter_trio_c_epsilon_and_gamma.md)
  * Guide 05: [Robustness to Outliers: SVR vs OLS vs RF](05_SVM_Family/02_Support_Vector_Regressor/05_robustness_to_outliers_svr_vs_ols_vs_rf.md)
  * Guide 06: [Production Pipeline & Latency Benchmarks](05_SVM_Family/02_Support_Vector_Regressor/06_production_pipeline_and_latency_benchmarks.md)

---

### 🚀 [06. Boosting Family Hub](06_Boosting_Family/README.md)
* **[01_AdaBoost_Classifier](06_Boosting_Family/01_AdaBoost_Classifier/README.md):**
  * Guide 01: [Sequential Boosting & Weak Learner Intuition](06_Boosting_Family/01_AdaBoost_Classifier/01_sequential_boosting_and_weak_learner_intuition.md)
  * Guide 02: [The AdaBoost Mathematics: Sample Weights & Alpha](06_Boosting_Family/01_AdaBoost_Classifier/02_the_adaboost_mathematics_sample_weights_and_alpha.md)
  * Guide 03: [Exponential Loss & Coordinate Descent Interpretation](06_Boosting_Family/01_AdaBoost_Classifier/03_exponential_loss_and_gradient_interpretation.md)
  * Guide 04: [Hyperparameter Tuning: Learning Rate & Estimators](06_Boosting_Family/01_AdaBoost_Classifier/04_hyperparameter_tuning_learning_rate_and_estimators.md)
  * Guide 05: [Production Pipeline & Latency Benchmarks](06_Boosting_Family/01_AdaBoost_Classifier/05_production_pipeline_and_latency_benchmarks.md)
* **[02_AdaBoost_Regressor](06_Boosting_Family/02_AdaBoost_Regressor/README.md):**
  * Guide 01: [AdaBoost Regression & Drucker's AdaBoost.R2 Algorithm](06_Boosting_Family/02_AdaBoost_Regressor/01_adaboost_regression_and_druckers_algorithm.md)
  * Guide 02: [Loss Functions: Linear vs Square vs Exponential](06_Boosting_Family/02_AdaBoost_Regressor/02_loss_functions_linear_square_exponential.md)
  * Guide 03: [Sample Reweighting & The Weighted Median Prediction Rule](06_Boosting_Family/02_AdaBoost_Regressor/03_sample_reweighting_and_weighted_median_prediction.md)
  * Guide 04: [Hyperparameter Optimization & Shrinkage](06_Boosting_Family/02_AdaBoost_Regressor/04_hyperparameter_optimization_and_shrinkage.md)
  * Guide 05: [Production Pipeline & Latency Benchmarks](06_Boosting_Family/02_AdaBoost_Regressor/05_production_pipeline_and_latency_benchmarks.md)
* **[03_Gradient_Boosting_Classifier](06_Boosting_Family/03_Gradient_Boosting_Classifier/README.md):**
  * Guide 01: [GBM Philosophy & Function-Space Gradient Descent](06_Boosting_Family/03_Gradient_Boosting_Classifier/01_gradient_boosting_machine_philosophy_and_function_space_descent.md)
  * Guide 02: [Mathematics of Pseudo-Residuals & Leaf Outputs](06_Boosting_Family/03_Gradient_Boosting_Classifier/02_the_mathematics_of_pseudo_residuals_and_leaf_output_values.md)
  * Guide 03: [Shrinkage, Learning Rate & Stochastic Subsampling](06_Boosting_Family/03_Gradient_Boosting_Classifier/03_shrinkage_learning_rate_and_stochastic_subsampling.md)
  * Guide 04: [Hyperparameter Tuning & Early Stopping](06_Boosting_Family/03_Gradient_Boosting_Classifier/04_hyperparameter_tuning_overfitting_and_early_stopping.md)
  * Guide 05: [Production Pipeline & Latency Benchmarks](06_Boosting_Family/03_Gradient_Boosting_Classifier/05_production_pipeline_and_latency_benchmarks.md)
* **[04_Gradient_Boosting_Regressor](06_Boosting_Family/04_Gradient_Boosting_Regressor/README.md):**
  * Guide 01: [Gradient Boosting Regression & Loss Functions](06_Boosting_Family/04_Gradient_Boosting_Regressor/01_gradient_boosting_regression_and_loss_functions.md)
  * Guide 02: [Regression Gradient Descent Algorithm & Leaf Updates](06_Boosting_Family/04_Gradient_Boosting_Regressor/02_the_regression_gradient_descent_algorithm_and_leaf_updates.md)
  * Guide 03: [Huber Loss & Quantile Loss for Robust Modeling](06_Boosting_Family/04_Gradient_Boosting_Regressor/03_huber_loss_and_quantile_loss_for_robust_modeling.md)
  * Guide 04: [Hyperparameter Optimization & Stochastic Subsampling](06_Boosting_Family/04_Gradient_Boosting_Regressor/04_hyperparameter_tuning_learning_rate_and_stochastic_subsampling.md)
  * Guide 05: [Production Pipeline & Latency Benchmarks](06_Boosting_Family/04_Gradient_Boosting_Regressor/05_production_pipeline_and_latency_benchmarks.md)
* **[05_Hist_Gradient_Boosting](06_Boosting_Family/05_Hist_Gradient_Boosting/README.md):**
  * Guide 01: [Histogram Binning & LightGBM Roots](06_Boosting_Family/05_Hist_Gradient_Boosting/01_histogram_binning_and_lightgbm_roots.md)
  * Guide 02: [HistGradientBoosting Classifier & Regressor Mechanics](06_Boosting_Family/05_Hist_Gradient_Boosting/02_hist_gradient_boosting_classifier_and_regressor_mechanics.md)
  * Guide 03: [Production Pipeline & Latency Benchmarks](06_Boosting_Family/05_Hist_Gradient_Boosting/03_production_pipeline_and_latency_benchmarks.md)

---

### 🤝 [07. Ensemble Methods Hub](07_Ensemble_Methods/README.md)
* **[01_Voting](07_Ensemble_Methods/01_Voting/README.md):**
  * Guide 01: [Hard vs. Soft Voting Principles & Condorcet's Theorem](07_Ensemble_Methods/01_Voting/01_hard_vs_soft_voting_principles.md)
  * Guide 02: [Production Pipeline & Latency Benchmarks](07_Ensemble_Methods/01_Voting/02_production_pipeline_and_latency_benchmarks.md)
* **[02_Stacking](07_Ensemble_Methods/02_Stacking/README.md):**
  * Guide 01: [Stacked Generalization & Out-Of-Fold Meta-Learning](07_Ensemble_Methods/02_Stacking/01_stacked_generalization_and_out_of_fold_meta_learning.md)
  * Guide 02: [Production Pipeline & Latency Benchmarks](07_Ensemble_Methods/02_Stacking/02_production_pipeline_and_latency_benchmarks.md)
* **[03_Bagging_and_Pasting](07_Ensemble_Methods/03_Bagging_and_Pasting/README.md):**
  * Guide 01: [Bootstrap Aggregating vs. Pasting & Random Subspaces](07_Ensemble_Methods/03_Bagging_and_Pasting/01_bootstrap_aggregating_vs_pasting_and_random_subspaces.md)
  * Guide 02: [Production Pipeline & Latency Benchmarks](07_Ensemble_Methods/03_Bagging_and_Pasting/02_production_pipeline_and_latency_benchmarks.md)
* **[04_Extra_Trees](07_Ensemble_Methods/04_Extra_Trees/README.md):**
  * Guide 01: [Extremely Randomized Trees Mechanics & Random Thresholds](07_Ensemble_Methods/04_Extra_Trees/01_extremely_randomized_trees_mechanics_and_random_thresholds.md)
  * Guide 02: [Production Pipeline & Latency Benchmarks](07_Ensemble_Methods/04_Extra_Trees/02_production_pipeline_and_latency_benchmarks.md)

---

### 📈 [08. Advanced Regression Hub](08_Advanced_Regression/README.md)
* **[Generalized Linear Models (GLM)](08_Advanced_Regression/Generalized_Linear_Models.md):** Poisson, Gamma, Tweedie, & **Negative Binomial** ($V(\mu) = \mu + \alpha\mu^2$).
* **[Robust Regression](08_Advanced_Regression/Robust_Regression.md):** Huber, RANSAC, and Theil-Sen regressors.
* **[Polynomial Regression](08_Advanced_Regression/Polynomial_Regression.md):** Non-linear basis expansion & Vandermonde structures.
* **[Quantile Regression](08_Advanced_Regression/Quantile_Regression.md):** Asymmetric pinball loss & heteroscedastic prediction bands.
* **[Gaussian Process Regression (GPR)](08_Advanced_Regression/Gaussian_Process_Regression.md):** Non-parametric Bayesian kernel regression with exact epistemic uncertainty.

---

### 🎯 [09. Advanced Classification Hub](09_Advanced_Classification/README.md)
* **[Multi-Class Strategies](09_Advanced_Classification/MultiClass_Strategies/README.md):** One-vs-Rest (OvR) and One-vs-One (OvO) meta-architectures.
* **[Probability Calibration](09_Advanced_Classification/Probability_Calibration/README.md):** Platt Scaling (Sigmoid) and Isotonic Regression for true frequentist confidence.

---

### 🧠 [10. Neural Networks Hub](10_Neural_Networks/README.md)
* **[Perceptron](10_Neural_Networks/Perceptron/README.md):** Rosenblatt's threshold learning rule and linear separability boundaries.
* **[Multi-Layer Perceptron (MLP)](10_Neural_Networks/MLPClassifier/README.md):** Universal Approximation Theorem, Backpropagation, Adam/L-BFGS optimization.

---

### 📐 [11. Discriminant Analysis Hub](11_Discriminant_Analysis/README.md)
* **Linear Discriminant Analysis (LDA):** Homoscedastic Gaussian Bayes, Fisher's between/within variance maximization.
* **Quadratic Discriminant Analysis (QDA):** Heteroscedastic class-specific covariance matrices and quadratic decision manifolds.

---

### 🎲 [12. Bayesian Regression Hub](12_Bayesian_Regression/README.md)
* **Bayesian Ridge Regression:** Spherical Gaussian priors, exact posterior distribution, MacKay evidence maximization.
* **Automatic Relevance Determination (ARD):** Component-wise hyperparameter priors for automated Bayesian feature sparsity.

---

### 📍 [13. Nearest Centroid Hub](13_Nearest_Centroid/README.md)
* **Nearest Centroid Classifier:** Rocchio prototype classification, sub-0.05ms edge serving latency, Nearest Shrunken Centroids (PAM).

---

### 🧪 [14. Partial Least Squares Hub](14_Partial_Least_Squares/README.md)
* **PLS Regression (PLSRegression):** Supervised bilinear latent component extraction under severe multicollinearity ($p \gg N$).
* **PLS Canonical (PLSCanonical):** Bidirectional canonical correlation maximization across multi-target response manifolds.

---

### ⚡ [15. Passive-Aggressive Suite Hub](15_Passive_Aggressive/README.md)
* **PAClassifier & PARegressor:** Online streaming margin updates, hinge loss analytical projection step, zero learning-rate tuning.

---

### ⚖️ [16. Cost-Sensitive Learning Hub](16_Cost_Sensitive_Learning/README.md)
* **Cost-Sensitive Classifier:** Asymmetric cost matrices, Bayes optimal decision thresholding $\tau^* = \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}$, class-weighted loss scaling.

---

### ⛓️ [17. Multi-Output & Chain Hub](17_Multi_Output/README.md)
* **MultiOutputClassifier & MultiOutputRegressor:** Independent parallel multi-target decomposition.
* **ClassifierChain & RegressorChain:** Autoregressive conditional dependency chains capturing cross-target covariances.

---

### 🥇 [18. Ordinal Classification Hub](18_Ordinal_Classification/README.md)
* **Ordinal Classifier:** Frank & Hall (2001) threshold decomposition, Quadratic Weighted Kappa (QWK) optimization.

---

### 📐 [19. Orthogonal Distance Regression Hub](19_Orthogonal_Distance_Regression/README.md)
* **Orthogonal Distance Regression (ODR):** Total Least Squares modeling observation noise in both covariates $\mathbf{X}$ and target $Y$.

---

### ⏳ [20. Survival Analysis Hub](20_Survival_Analysis/README.md)
* **Cox Proportional Hazards & AFT:** Semi-parametric right-censoring handling, hazard functions $h(t)$, Harrell's Concordance Index (C-index).

---

### 🎯 [21. Learning-to-Rank Hub](21_Ranking_LTR/README.md)
* **Ranking / LTR:** Pointwise, Pairwise RankNet, and Listwise LambdaMART ($\lambda$-gradient tree boosting) for direct NDCG@K optimization.

---

### 🧩 [22. Factorization Machines Hub](22_Factorization_Machines/README.md)
* **Factorization Machines (FM & FFM):** Steffen Rendle 2nd-order latent embeddings, $\mathcal{O}(k \cdot p)$ linear-time interaction trick for sparse data.

---

## 🏛️ Architecture: Algorithm Families vs. Supporting Concepts

```text
SUPERVISED ALGORITHM FAMILIES (f: X -> Y)
├── 01. Linear Models (Linear, Logistic, Ridge, Lasso, ElasticNet, SGD)
├── 02. Distance-Based (KNN Classifier & Regressor)
├── 03. Probabilistic (Gaussian, Multinomial, Bernoulli Naive Bayes)
├── 04. Tree-Based (Decision Trees, Random Forest)
├── 05. Support Vector Machines (SVC, SVR)
├── 06. Boosting Family (AdaBoost, Gradient Boosting, HistGBM, XGBoost, LightGBM, CatBoost)
├── 07. Ensemble Methods (Voting, Stacking, Blending, Bagging, Extra Trees)
├── 08. Advanced Regression (GLM: Poisson/Gamma/Tweedie/NegBin, Robust, Poly, Quantile, GPR)
├── 09. Advanced Classification (Multi-Class OvR/OvO, Probability Calibration)
├── 10. Neural Networks (Perceptron, MLPClassifier, MLPRegressor)
├── 11. Discriminant Analysis (LDA, QDA)
├── 12. Bayesian Regression (Bayesian Ridge, ARD Regression)
├── 13. Nearest Centroid Classifier
├── 14. Partial Least Squares (PLS Regression, PLS Canonical)
├── 15. Passive-Aggressive Algorithms (PAClassifier, PARegressor)
├── 16. Cost-Sensitive Learning (Asymmetric loss, Bayes optimal thresholding)
├── 17. Multi-Output Models (MultiOutput, Classifier Chains, Regressor Chains)
├── 18. Ordinal Classification (Frank & Hall threshold decomposition)
├── 19. Orthogonal Distance Regression (ODR / Errors-in-Variables)
├── 20. Survival Analysis (Cox Proportional Hazards, AFT)
├── 21. Learning-to-Rank (Pointwise, Pairwise, Listwise LambdaMART)
└── 22. Factorization Machines (FM, FFM)

SUPPORTING MACHINE LEARNING CONCEPTS (Methodological Infrastructure)
├── Cross-Validation (K-Fold, StratifiedKFold, GroupKFold, TimeSeriesSplit)
├── Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV, Optuna BayesOpt)
├── Feature Engineering & Transformation (PowerTransformer, QuantileTransformer, Splines)
├── Feature Selection (SelectKBest, RFE, Permutation Importance, SHAP values)
├── Imbalanced Learning (SMOTE, ADASYN, NearMiss, Class Weighting)
├── Data Hygiene & Auditing (Zero-leakage split, cardinality analysis, missing value imputers)
└── Production Deployment (Pipeline serialization, sub-50ms latency smoke testing)
```


