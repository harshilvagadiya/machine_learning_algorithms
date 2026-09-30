# 🚀 Gradient Boosting Classifier Master Index

Welcome to the Gradient Boosting Classifier memory notes. Covers function-space gradient descent and binomial deviance loss.

---

## 📚 Guides in this Folder

1. [**01. GBM Philosophy & Function-Space Gradient Descent**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/01_gradient_boosting_machine_philosophy_and_function_space_descent.md)
   - Parameter-space vs. function-space optimization
   - Fitting decision trees to negative gradients
2. [**02. Mathematics of Pseudo-Residuals & Leaf Outputs**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/02_the_mathematics_of_pseudo_residuals_and_leaf_output_values.md)
   - Binomial deviance loss derivation
   - Newton-Raphson leaf output formula $\gamma_{jm}$
3. [**03. Shrinkage, Learning Rate & Stochastic Subsampling**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/03_shrinkage_learning_rate_and_stochastic_subsampling.md)
   - Stochastic row subsampling (`subsample < 1.0`)
   - Feature subsampling (`max_features < 1.0`)
4. [**04. Hyperparameter Tuning & Early Stopping**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/04_hyperparameter_tuning_overfitting_and_early_stopping.md)
   - Early stopping on internal validation split
   - Essential hyperparameter grid
5. [**05. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/05_production_pipeline_and_latency_benchmarks.md)
   - Inference latency profile ($< 0.7$ ms)
   - Fraud scoring production microservice
