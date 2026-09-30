# 📉 Gradient Boosting Regressor Master Index

Welcome to the Gradient Boosting Regressor architectural memory notes. Covers differentiable loss minimization, pseudo-residual derivation, and robust Huber loss.

---

## 📚 Guides in this Folder

1. [**01. Gradient Boosting Regression & Loss Functions**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/01_gradient_boosting_regression_and_loss_functions.md)
   - Objective functions: squared error, absolute error, Huber, quantile
   - The squared error special case (iterative residual chasing)
2. [**02. Regression Gradient Descent Algorithm & Leaf Updates**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/02_the_regression_gradient_descent_algorithm_and_leaf_updates.md)
   - Full mathematical algorithm derivation
   - Optimal constant initialization $F_0(x)$ and terminal leaf value updates
3. [**03. Huber Loss & Quantile Loss for Robust Modeling**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/03_huber_loss_and_quantile_loss_for_robust_modeling.md)
   - Mathematical formulation of Huber threshold $\delta$
   - Training 10th, 50th, and 90th percentile prediction interval bands
4. [**04. Hyperparameter Optimization & Stochastic Subsampling**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/04_hyperparameter_tuning_learning_rate_and_stochastic_subsampling.md)
   - Interaction between learning rate $\eta$, `max_depth`, and subsample
   - Production GridSearchCV recipe
5. [**05. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/05_production_pipeline_and_latency_benchmarks.md)
   - Latency benchmarks ($< 1.0$ ms) and model artifact size
   - Real-time pricing prediction microservice implementation
