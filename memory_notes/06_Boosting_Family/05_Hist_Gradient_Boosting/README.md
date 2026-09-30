# 📊 HistGradientBoosting Master Index

Welcome to the HistGradientBoosting architectural memory notes. Covers histogram-based continuous binning, native missing value support, and LightGBM optimizations.

---

## 📚 Guides in this Folder

1. [**01. Histogram Binning & LightGBM Roots**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/05_Hist_Gradient_Boosting/01_histogram_binning_and_lightgbm_roots.md)
   - 256-integer feature quantization (`uint8`)
   - $\mathcal{O}(N)$ histogram aggregation and histogram subtraction trick
   - Native missing value handling
2. [**02. HistGradientBoosting Classifier & Regressor Mechanics**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/05_Hist_Gradient_Boosting/02_hist_gradient_boosting_classifier_and_regressor_mechanics.md)
   - Objectives, automatic early stopping, and monotonic constraints
   - Minimalist enterprise implementation
3. [**03. Production Pipeline & Latency Benchmarks**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/05_Hist_Gradient_Boosting/03_production_pipeline_and_latency_benchmarks.md)
   - Sub-0.4ms inference benchmarks
   - High-throughput web inference microservice
