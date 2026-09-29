# 🛡️ Ridge Regression ($L_2$ Regularization): Projects & Production Suite

Yeh folder **Ridge Regression ($L_2$ Regularization)** ke end-to-end practical machine learning projects ka production-ready hub hai.

---

## 🗂️ Project Directory Structure

```text
RidgeRegression/
├── README.md                                          # Documentation & Architecture Guide
│
├── 📓 Notebooks:
│   └── House_Price_Prediction_Ridge.ipynb             # Ames Housing 12-Step Production Pipeline
│
├── 📂 data/                                           # Clean Structured Datasets
│   └── house_price/                                   # Kaggle House Prices (train.csv, test.csv)
│
└── 📦 production_models/                              # Serialized Production Artifacts
    └── house_price_ridge_pipeline.joblib              # Full End-to-End Pipeline (Preprocessor + Ridge)
```

---

## 🔬 Core Project: House Price Prediction with Many Features

- **Dataset:** Kaggle Ames Housing Dataset (1,460 rows × 81 columns)
- **Target Variable ($y$):** `SalePrice` (Transformed with `np.log1p` to correct skewness from $1.88 \to 0.12$)
- **Feature Space:** 21 Continuous features + 58 Categorical features $\to$ One-Hot expanded to **333 predictor dimensions**.
- **The Core Problem Solved:** Severe Multicollinearity ($r > 0.80$ across multiple pairs like `GarageCars` & `GarageArea`, `TotalBsmtSF` & `1stFlrSF`) where standard OLS weights explode.

---

## 📐 Ridge Mathematical Formulation

$$J(\theta) = \text{MSE}(\theta) + \alpha \sum_{j=1}^{p} \theta_j^2$$

Closed-form analytical solution:
$$\hat{\beta}_{\text{ridge}} = (X^T X + \alpha I)^{-1} X^T y$$

- Adding $\alpha I$ to the diagonal makes $(X^T X + \alpha I)$ strictly invertible even when $p > n$ or features are collinear.
- Penalty forces weights closer to zero (Shrinkage) without dropping features.

---

## 🏆 Model Benchmark & Performance

| Metric | OLS Linear Regression | Ridge Regression ($L_2$ Regularized) | Impact / Note |
| :--- | :---: | :---: | :--- |
| **Optimal Regularization ($\alpha$)** | $\alpha = 0$ | **$\alpha = 11.51$** | Tuned via `RidgeCV` |
| **Train $R^2$ Score** | 95.22% | 92.44% | Ridge controls overfitting |
| **Test $R^2$ Score** | 89.48% | **89.02%** | High generalization stability |
| **Max Absolute Weight** | $1.6767$ | **$0.1202$** | **92.8% reduction in weight explosion!** |
| **Serialized Artifact** | N/A | `house_price_ridge_pipeline.joblib` | 23.7 KB deployable pipeline |

---

## 🚀 Execution & Usage

To load and run live predictions using the serialized production artifact:

```python
import joblib
import numpy as np
import pandas as pd

# 1. Load the serialized pipeline
pipeline = joblib.load('production_models/house_price_ridge_pipeline.joblib')

# 2. Feed raw house attributes (Pandas DataFrame)
sample_house = pd.read_csv('data/house_price/train.csv').drop(columns=['Id', 'SalePrice']).iloc[0:1]

# 3. Predict in dollar scale
predicted_log_price = pipeline.predict(sample_house)[0]
predicted_price = np.expm1(predicted_log_price)
print(f"Predicted House Value: ${predicted_price:,.2f}")
```

