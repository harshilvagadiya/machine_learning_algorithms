# 🥞 Stacking (Stacked Generalization) Master Index

Welcome to the Stacking architectural memory notes. Stacking provides an optimal framework for meta-learning across heterogeneous estimators.

---

## 📚 Guides in this Folder

1. [**01. Stacked Generalization & Out-Of-Fold Meta-Learning**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/02_Stacking/01_stacked_generalization_and_out_of_fold_meta_learning.md)
   - Why naive stacking causes disastrous data leakage
   - K-Fold Out-Of-Fold (OOF) cross-validation mechanics
   - Mathematical formulation for StackingRegressor and StackingClassifier
   - Simple and enterprise production implementation patterns

2. [**02. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/07_Ensemble_Methods/02_Stacking/02_production_pipeline_and_latency_benchmarks.md)
   - Real-time online latency decomposition
   - Why Level-1 meta-learners must be strictly regularized linear models
   - Passthrough tradeoffs
   - Production inference service implementation
