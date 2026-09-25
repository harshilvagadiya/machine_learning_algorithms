# 🌐 Common Linear Family: 01 - Universal Enterprise Libraries Import

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho jab tum cooking karne kitchen mein jaate ho, to pehle saare masale, tel, bartan aur chammach ek jagah nikaal kar rakhte ho.  
> Machine Learning mein Linear-family ka koi bhi model banana ho (Linear, Logistic, Ridge, Lasso, ElasticNet, SGD), **90% libraries wahi same hoti hain!**  
> Baar-baar alag-alag import yaad rakhne ki zaroorat nahi hai. Yeh ek universal block copy karo aur shuru ho jao!

---

## 🧭 The Core Library Arsenal

```
                    UNIVERSAL LINEAR-FAMILY TOOLKIT
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
CORE DATA & MATH           DATA VISUALIZATION        SCIKIT-LEARN PIPELINES
- numpy (Arrays, logs)     - matplotlib (Plots)      - Pipeline, ColumnTransformer
- pandas (DataFrames)      - seaborn (Heatmaps)      - StandardScaler, OneHotEncoder
                                                     - SimpleImputer, train_test_split
                                                     - joblib (Model Serialization)
```

---

## 📦 Kyun Har Library Zaroori Hai? (Dumb Student Breakdown)

1. **`warnings.filterwarnings('ignore')`:**  
   Python ke be-wajah red alert warning messages ko chup karata hai taaki notebook clean dikhe.
2. **`numpy` & `pandas`:**  
   Data loading, matrix calculations, aur target log-transformations (`np.log1p`, `np.expm1`) ke liye.
3. **`matplotlib` & `seaborn`:**  
   Target distribution ka bell-curve dekhne aur Multicollinearity heatmap dekhne ke liye.
4. **`ColumnTransformer` & `Pipeline`:**  
   Preprocessors (Scaler + Encoder) aur Model ko ek sealed dabbe mein pack karne ke liye taaki Data Leakage na ho.
5. **`joblib`:**  
   Trained model ko file (`.joblib`) mein save karke production API mein bhejne ke liye.

---

## 📋 Copy-Paste Boilerplate: Universal Linear-Family Import Block
*(Isko kisi bhi linear algorithm notebook ke Cell 1 mein paste kar do)*

```python
# ==============================================================================
# 🌐 STEP 1: UNIVERSAL ENTERPRISE LIBRARIES IMPORT (LINEAR-FAMILY)
# ==============================================================================
import warnings
from pathlib import Path

# 1. Mathematical Computing & Data Handling
import numpy as np
import pandas as pd

# 2. Suppress runtime warnings for clean presentation
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# 3. Data Visualization & Theming
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10

# 4. Scikit-Learn: Preprocessing, Imputation & Splitting
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 5. Linear Family Estimators (Pick whichever is needed)
from sklearn.linear_model import (
    LinearRegression,      # 01. Baseline OLS Regression
    LogisticRegression,    # 02. Interpretable Classification
    Ridge, RidgeCV,        # 03. L2 Regularization (Multicollinearity)
    Lasso, LassoCV,        # 04. L1 Regularization (Feature Selection)
    ElasticNet, ElasticNetCV, # 05. Combined L1+L2 Regularization
    SGDRegressor,          # 06. Stochastic Gradient Descent Regression
    SGDClassifier          # 07. Stochastic Gradient Descent Classification
)

# 6. Evaluation Metrics
from sklearn.metrics import (
    r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, # Regression
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report # Classification
)

# 7. Production Model Serialization
import joblib

print("✅ Universal Linear-Family Enterprise Suite imported successfully.")
```

