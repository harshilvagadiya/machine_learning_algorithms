# 🛡️ Ridge Regression Memory Notes: 07 - Production Pipeline & Live Inference

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho tum Amazon se phone order karte ho. Kya Amazon phone alag dabba mein, charger doosre packet mein, aur cover teesre packet mein bhejta hai?  
> Nahi! Sab ek hi sealed box mein pack hokar aata hai.  
> Machine Learning mein agar tum Scaler alag file mein rakhoge, One-Hot Encoder alag script mein, aur Ridge Model alag, to Production Software Engineer pagal ho jayega aur live app crash ho jayegi!  
> **Scikit-Learn Pipeline** wahi sealed packaging box hai jo sabko ek saath jod deta hai!

---

## 🧭 The Production Architecture

```
                       END-TO-END RIDGE PIPELINE
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
FEATURE PREPROCESSING                                 MODEL ESTIMATOR
- Numeric: SimpleImputer + StandardScaler             - Ridge(alpha=11.51)
- Categoric: SimpleImputer + OneHotEncoder            - Regularized Prediction
                                   │
                                   ▼
                       SERIALIZATION (JOBLIB)
                       house_price_ridge_pipeline.joblib
                                   │
                                   ▼
                       LIVE API / FASTAPI INFERENCE
                       Input: Raw JSON Dict -> Output: $147,058
```

---

## 🏭 Step 1: Master Preprocessing (ColumnTransformer)

Real-world datasets mein do tarah ke columns hote hain:
1. **Numeric (Continuous):** Missing value ko Median se bharo + `StandardScaler` lagao.
2. **Categorical:** Missing value ko `'None'` se bharo + `OneHotEncoder(drop='first')` lagao.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 1. Numerical Pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()) # Mandatory for Ridge!
])

# 2. Categorical Pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 3. Master Preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, continuous_features),
    ('cat', cat_pipeline, categorical_features)
])
```

---

## 📦 Step 2: Master Pipeline Assembly & Serialization

```python
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge

# 1. Assemble Full End-to-End Pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=11.51, random_state=42))
])

# 2. Single Fit on Raw Data (Zero Leakage!)
full_pipeline.fit(X_train, y_train_log)

# 3. Save to Production Folder
model_path = Path('production_models/house_price_ridge_pipeline.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(full_pipeline, model_path)
print(f"✅ Serialized Pipeline Size: {model_path.stat().st_size / 1024:.2f} KB")
```

---

## 🧪 Step 3: Live Inference Smoke Test (In Real Time)

Production server par raw input aayega (e.g. Website par customer ne details dali).  
Aapko bas model load karna hai aur seedha predict karna hai:

```python
import joblib
import numpy as np
import pandas as pd

# 1. Load Serialized Pipeline
pipeline = joblib.load('production_models/house_price_ridge_pipeline.joblib')

# 2. Raw Unseen House Input
sample_house = pd.DataFrame([{
    'OverallQual': 7,
    'GrLivArea': 1710,
    'GarageCars': 2,
    'GarageArea': 548,
    'TotalBsmtSF': 856,
    '1stFlrSF': 856,
    'YearBuilt': 2003,
    'Neighborhood': 'CollgCr',
    # ... baki saare raw features
}])

# 3. Predict Log Price and Invert via expm1
pred_log = pipeline.predict(sample_house)[0]
final_dollar_price = np.expm1(pred_log)

print(f"🏠 Real-time Valuation: ${final_dollar_price:,.2f}")
```

---

## 📋 The 5-Point Production Deployment Checklist

1. [x] **Zero Data Leakage:** Preprocessing statistics sirf training split se aani chahiye.
2. [x] **StandardScaler Included:** Har continuous feature scale ho kar Ridge mein jaana chahiye.
3. [x] **`handle_unknown='ignore'`:** Agar production mein naya category value aa jaye jo training mein nahi tha, to code error na de!
4. [x] **Log Target Inverse:** Target ko agar `np.log1p` kiya tha to return karte waqt `np.expm1` lagana mat bhoolo!
5. [x] **Single Artifact:** Poora pipeline (imputer + scaler + encoder + ridge) ek hi `.joblib` file mein hona chahiye.

---

## 📋 Copy-Paste Boilerplate: Reusable Enterprise Production Pipeline Engine
*(Isko copy karke kisi bhi enterprise regression project mein laga do, yeh full pipeline train karega, `.joblib` save karega, aur real-time predictions dega)*

```python
# ==============================================================================
# COPY-PASTE SNIPPET: COMPLETE ENTERPRISE PRODUCTION PIPELINE BUILDER
# ==============================================================================
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge

class RidgeProductionEngine:
    def __init__(self, continuous_cols, categorical_cols, alpha=10.0, is_target_log=True):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.alpha = alpha
        self.is_target_log = is_target_log
        self.pipeline = None

    def build_and_fit(self, X_train, y_train):
        """Constructs end-to-end ColumnTransformer + Ridge pipeline and fits on train."""
        # 1. Pipelines for numerical & categorical
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # 2. Master ColumnTransformer
        preprocessor = ColumnTransformer([
            ('num', num_pipe, self.continuous_cols),
            ('cat', cat_pipe, self.categorical_cols)
        ])
        
        # 3. Full Pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ridge', Ridge(alpha=self.alpha, random_state=42))
        ])
        
        # 4. Single Fit (Zero Leakage)
        self.pipeline.fit(X_train, y_train)
        print("✅ Production Pipeline fitted successfully with zero data leakage.")
        return self

    def save(self, filepath="production_models/ridge_pipeline.joblib"):
        """Serializes the entire pipeline to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"📦 Serialized pipeline saved to: {path} ({path.stat().st_size / 1024:.2f} KB)")

    @classmethod
    def load_and_predict(cls, filepath, raw_input_df, is_target_log=True):
        """Loads pipeline from disk and predicts on raw input DataFrame."""
        pipeline = joblib.load(filepath)
        preds = pipeline.predict(raw_input_df)
        if is_target_log:
            preds = np.expm1(preds)
        return preds

# --- USAGE EXAMPLE ---
# engine = RidgeProductionEngine(continuous_cols=num_cols, categorical_cols=cat_cols, alpha=11.51)
# engine.build_and_fit(X_train, y_train_log)
# engine.save("production_models/house_price_ridge_pipeline.joblib")
#
# # Real-time inference on new observation:
# live_preds = RidgeProductionEngine.load_and_predict(
#     "production_models/house_price_ridge_pipeline.joblib", 
#     raw_input_df=X_test.iloc[0:1], 
#     is_target_log=True
# )
# print("Live Prediction:", live_preds[0])
```


