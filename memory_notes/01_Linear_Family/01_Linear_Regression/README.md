# 📈 Supervised Regression Master Memory Notes

Yeh folder Supervised Regression ka complete step-by-step master handbook hai. Har topic ko basic se lekar advanced industry level tak explain kiya gaya hai.

---

## 🗺️ Regression Roadmap (Step-by-Step)

| Step | File Name | Core Focus & Techniques |
| :---: | :--- | :--- |
| **01** | [01_data_loading_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/01_data_loading_guide.md) | Local CSV, Excel (`.xlsx` via `openpyxl`), Kaggle 3-line pattern, `latin-1` encoding, memory optimization |
| **02** | [02_data_types_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/02_data_types_guide.md) | `df.dtypes`, auto-split `num_cols`/`cat_cols`, 4 traps (fake string, fake number, `errors='coerce'`, dates) |
| **03** | [03_missing_values_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/03_missing_values_guide.md) | Missing value decision tree, bar charts, SimpleImputer (mean/median/mode), zero-leakage pipeline |
| **04** | [04_duplicate_rows_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/04_duplicate_rows_guide.md) | Duplicate detection, `.count()` vs `.sum()` trap, visual pie chart, data leakage prevention |
| **05** | [05_target_analysis_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/05_target_analysis_guide.md) | 5-point summary, 3-panel visual diagnostics (Hist/KDE, Boxplot, Q-Q), Skewness check, `log1p`/`expm1` |
| **06** | [06_outlier_detection_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/06_outlier_detection_guide.md) | Outlier decision tree, IQR 1.5 formula vs Z-score, Winsorization / Capping (`np.clip`), Pearson correlation ranking |
| **07** | [07_encoding_and_feature_engineering_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/07_encoding_and_feature_engineering_guide.md) | One-Hot (`drop_first=True`), Ordinal, Multi-label (`.str.get_dummies()`), High Cardinality Top-N, Multicollinearity/VIF |
| **08** | [08_model_training_and_evaluation_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/regression/08_model_training_and_evaluation_guide.md) | Train/Test Split, Scaling (Standard/MinMax/Robust), OLS vs Ridge ($L_2$) vs Lasso ($L_1$), MAE, RMSE, $R^2$, Residual diagnostics, `joblib` saving |

