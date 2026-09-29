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
