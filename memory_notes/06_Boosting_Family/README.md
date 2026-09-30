# 🚀 Boosting Family Architecture & Memory Notes

The Boosting Family consists of iterative algorithms that convert committees of weak base hypotheses into a single high-precision predictor by sequentially focusing capacity on residual prediction errors.

---

## 🏛️ Boosting Family Taxonomy

```
                      ┌──────────────────────────────────────────────┐
                      │               BOOSTING FAMILY                │
                      └──────────────────────┬───────────────────────┘
                                             │
      ┌──────────────────────────────────────┴──────────────────────────────────────┐
      │                                                                             │
┌─────▼──────────┐                                                           ┌──────▼─────────┐
│ 01. ADABOOST   │                                                           │ 02. GRADIENT   │
│ Adaptive       │                                                           │     BOOSTING   │
│ Sample Weights │                                                           │ Function-Space │
│ Exponential L. │                                                           │ Pseudo-Resids  │
└─────┬──────────┘                                                           └──────┬─────────┘
      │                                                                             │
 ┌────┴──────────────────┐                                                     ┌────┴──────────────────┐
 │                       │                                                     │                       │
┌▼──────────────┐ ┌──────▼───────┐                                            ┌▼──────────────┐ ┌──────▼───────┐
│ 01_Classifier │ │ 02_Regressor │                                            │ 03_Classifier │ │ 04_Regressor │
│ AdaBoost.M1   │ │ AdaBoost.R2  │                                            │ Log-Loss Dev. │ │ Huber & MSE  │
└───────────────┘ └──────────────┘                                            └──────┬────────┘ └──────────────┘
                                                                                     │
                                                                              ┌──────▼────────┐
                                                                              │ 05_Hist_Grad  │
                                                                              │ 256-Bin Quant │
                                                                              │ Native NaNs   │
                                                                              └───────────────┘
```

---

## 📚 Section Breakdown & Navigation

| Module | Core Theoretical Concept | Primary Benefit | Guide Index |
| :--- | :--- | :--- | :--- |
| [**01. AdaBoost Classifier**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/README.md) | Sample reweighting via exponential loss: $w^{(m+1)} = w^{(m)} e^{-\alpha y h(x)}$. | Ultra-fast inference with decision stumps; elegant margins. | [01_AdaBoost_Classifier/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/01_AdaBoost_Classifier/README.md) |
| [**02. AdaBoost Regressor**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/README.md) | Drucker's AdaBoost.R2 algorithm with weighted median prediction. | Robust continuous target boosting without extreme outlier inflation. | [02_AdaBoost_Regressor/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/02_AdaBoost_Regressor/README.md) |
| [**03. Gradient Boosting Classifier**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/README.md) | Function-space gradient descent on binomial deviance log-loss. | Industry gold standard accuracy on structured tabular datasets. | [03_Gradient_Boosting_Classifier/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/03_Gradient_Boosting_Classifier/README.md) |
| [**04. Gradient Boosting Regressor**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/README.md) | Sequential residual fitting with squared error, Huber, and quantile loss. | Extreme forecasting precision; calibrated prediction intervals. | [04_Gradient_Boosting_Regressor/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/04_Gradient_Boosting_Regressor/README.md) |
| [**05. HistGradientBoosting**](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/05_Hist_Gradient_Boosting/README.md) | LightGBM-style 256-integer histogram binning with native missing values. | $10\times-100\times$ faster training; zero imputation boilerplate. | [05_Hist_Gradient_Boosting/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/06_Boosting_Family/05_Hist_Gradient_Boosting/README.md) |
