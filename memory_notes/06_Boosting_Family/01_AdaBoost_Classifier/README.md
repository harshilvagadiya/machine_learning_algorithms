# ⚡ AdaBoost Classifier Master Index

Welcome to the AdaBoost Classifier memory notes. AdaBoost is the pioneer sequential boosting algorithm.

---

## 📚 Guides in this Folder

1. [**01. Sequential Boosting & Weak Learner Intuition**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/01_sequential_boosting_and_weak_learner_intuition.md)
   - Sequential error focusing vs. parallel bootstrap aggregation
   - Why decision stumps (`max_depth=1`) are the ideal weak learner
2. [**02. The Mathematics of Sample Weights & Estimator Weight $\alpha$**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/02_the_adaboost_mathematics_sample_weights_and_alpha.md)
   - Discrete AdaBoost.M1 algorithm derivation
   - Weight update formula $w^{(m+1)} = w^{(m)} e^{-\alpha y h(x)}$
3. [**03. Exponential Loss & The Gradient Interpretation**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/03_exponential_loss_and_gradient_interpretation.md)
   - Coordinate descent on exponential loss $L(y, F(x)) = e^{-y F(x)}$
   - Why AdaBoost is hyper-sensitive to label noise and outliers
4. [**04. Hyperparameter Tuning: Learning Rate & Estimators**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/04_hyperparameter_tuning_learning_rate_and_estimators.md)
   - Shrinkage rate $\eta$ and trade-off with `n_estimators`
   - Complete GridSearchCV tuning script
5. [**05. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/05_production_pipeline_and_latency_benchmarks.md)
   - Ultra-fast $\mathcal{O}(M)$ inference latency ($< 0.2$ ms)
   - Production serving and serialization
