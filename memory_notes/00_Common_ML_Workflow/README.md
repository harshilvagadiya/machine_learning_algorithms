# 🌐 00. Common Linear-Family Memory Notes (Universal Foundation)

> **The Golden Law of the Linear Family:**  
> Chahe aap **Linear Regression**, **Logistic Regression**, **Ridge**, **Lasso**, **ElasticNet**, ya **SGD** chala rahe hon — **data preprocessing, hygiene, splitting aur scaling ka 80% kaam har jagah bilkul identical hota hai!**  
> Yeh folder un sabhi universal, non-negotiable steps ka single source of truth hai. Ek baar yahan seekh lo, saare linear algorithms aasaan ho jayenge!

---

## 🧭 The 7 Universal Steps of Any Linear-Family Project

```
[Raw Data] 
    │
    ▼ 
(Step 1: Libraries) ──► (Step 2: Ingestion & 3-Sec Audit) ──► (Step 3: Feature Segregation)
                                                                       │
                                                                       ▼
(Step 6: Target Profiler) ◄── (Step 5: Train-Test Split) ◄── (Step 4: Missing & Duplicates)
           │
           ▼
(Step 7: Master ColumnTransformer: Scaling + OneHot) ──► [Ready for Any Linear Model!]
```

---

## 📚 The 7 Universal Guides (Basic to Advanced)

| Guide # | Document Title | What You Will Learn (In Bhai Language) |
| :---: | :--- | :--- |
| **01** | [01_universal_libraries_import_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/01_universal_libraries_import_guide.md) | Kitchen ke masale analogy: 100% clean, warning-free modular import block. |
| **02** | [02_data_ingestion_and_3sec_audit_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/02_data_ingestion_and_3sec_audit_guide.md) | Doctor ka BP check analogy: `Path` safety, Latin-1 encoding trap, shape & memory audit. |
| **02b** | [02b_data_sanitization_and_string_cleaning_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/02b_data_sanitization_and_string_cleaning_guide.md) | Kela chhilka analogy: Universal string-to-numeric parsing (`$`, `₹`, `,`, `m²`, `km`, `%` stripping & `coerce`). |
| **03** | [03_feature_segregation_and_cardinality_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/03_feature_segregation_and_cardinality_guide.md) | 3 Balti (Buckets) rule: Continuous, Categorical, Drop/ID columns & Cardinality threshold ($\le 15$). |
| **04** | [04_missing_values_and_duplicates_hygiene_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/04_missing_values_and_duplicates_hygiene_guide.md) | Duplicate photocopy trap, MCAR vs. MNAR (Amenity absence), 'None' category vs. Median. |
| **05** | [05_train_test_split_and_zero_leakage_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/05_train_test_split_and_zero_leakage_guide.md) | Exam paper leak analogy: Split BEFORE scaling/imputation, Regression vs. Classification split rules. |
| **06** | [06_target_analysis_and_transformation_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/06_target_analysis_and_transformation_guide.md) | Hero diagnostics: Regression `np.log1p` skewness correction vs. Classification Zero-Rule baseline accuracy trap. |
| **07** | [07_master_columntransformer_preprocessing_guide.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/07_master_columntransformer_preprocessing_guide.md) | Assembly juice factory analogy: `StandardScaler` + `OneHotEncoder(drop='first')` zero-leakage pipeline. |

---

## 🚀 Master Universal Boilerplate: Steps 1 Through 7 in One Block
*(Isko copy karke kisi bhi linear-family notebook ke starting cells mein run kar do — data load, audit, clean, split aur scale hokar model ready ho jayega!)*

```python
# ==============================================================================
# 🌐 MASTER FOUNDATION: STEPS 1 TO 7 FOR ANY LINEAR-FAMILY ALGORITHM
# ==============================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# 1. LOAD DATASET
# file_path = "your_data.csv"
# df = pd.read_csv(file_path, encoding='utf-8')
# target_col = "SalePrice"
# is_classification = False
# apply_log_target = True

# 2. DROP DUPLICATES (PREVENTS LEAKAGE)
df = df.drop_duplicates().reset_index(drop=True)

# 3. FEATURE SEGREGATION (Cardinality Threshold = 15)
cardinality_threshold = 15
cont_features, cat_features, drop_features = [], [], []

for col in df.columns:
    if col == target_col:
        continue
    if col.lower() in ['id', 'id_col', 'index', 'unnamed: 0']:
        drop_features.append(col)
        continue
    if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= cardinality_threshold:
        cat_features.append(col)
    else:
        cont_features.append(col)

print(f"Segregation: {len(cont_features)} Continuous, {len(cat_features)} Categorical, {len(drop_features)} Dropped")

# 4. FEATURE MATRIX & TARGET SPLIT
X = df.drop(columns=[target_col] + drop_features)
y = np.log1p(df[target_col]) if (apply_log_target and not is_classification) else df[target_col]

stratify_arg = y if is_classification else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=stratify_arg
)

# 5. MASTER PREPROCESSING (ZERO LEAKAGE COLUMNTRANSFORMER)
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Mandatory for all Linear Models!
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, cont_features),
    ('cat', cat_pipeline, cat_features)
])

# 6. FIT ON TRAIN ONLY, TRANSFORM TEST
X_train_final = preprocessor.fit_transform(X_train)
X_test_final  = preprocessor.transform(X_test)

print("=" * 70)
print(f"✅ READY FOR LINEAR MODEL TRAINING! Dimensions: {X_train_final.shape[1]} features")
print(f"   X_train: {X_train_final.shape} | X_test: {X_test_final.shape}")
print("=" * 70)
```

