# 📉 AdaBoost Regressor Master Index

Welcome to the AdaBoost Regressor architectural memory notes. Covers Drucker's algorithm (AdaBoost.R2) and weighted median aggregation.

---

## 📚 Guides in this Folder

1. [**01. AdaBoost Regression & Drucker's Algorithm**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/01_adaboost_regression_and_druckers_algorithm.md)
   - Harris Drucker's formulation (AdaBoost.R2)
   - Relative error scaling by maximum error $D$
   - Weighted median prediction derivation
2. [**02. Loss Functions: Linear vs. Square vs. Exponential**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/02_loss_functions_linear_square_exponential.md)
   - Mathematical error transformation curves
   - Outlier sensitivity across different loss functions
3. [**03. Sample Reweighting & Weighted Median Prediction**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/03_sample_reweighting_and_weighted_median_prediction.md)
   - Proof of weight dampening for low-error samples
   - Step-by-step sorting and weighted median calculation
4. [**04. Hyperparameter Optimization & Shrinkage**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/04_hyperparameter_optimization_and_shrinkage.md)
   - Tuning tree depth, learning rate $\eta$, and loss metric
   - Production GridSearchCV recipe
5. [**05. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/05_production_pipeline_and_latency_benchmarks.md)
   - Real-world latency performance ($< 0.5$ ms)
   - Microservice deployment architecture
