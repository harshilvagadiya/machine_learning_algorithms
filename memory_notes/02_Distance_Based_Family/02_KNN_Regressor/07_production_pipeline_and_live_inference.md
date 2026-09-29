# Guide 07: Production Pipeline, Serialization & Live Inference

> **Note on Workflow:**
> For general pipeline serialization patterns, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details **production deployment and real-time inference for KNN Regression**.

---

## 1. Production Pipeline Architecture

```python
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

# 1. Modular Preprocessor
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # CRITICAL FOR DISTANCE CALCULATION
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, cont_cols),
    ('cat', cat_pipeline, cat_cols)
])

# 2. Complete Model Pipeline
knn_reg_pipeline = Pipeline([
    ('prep', preprocessor),
    ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance', p=1))
])

# 3. Fit & Serialize
knn_reg_pipeline.fit(X_train, y_train)
joblib.dump(knn_reg_pipeline, 'production_models/knn_regressor_pipeline.joblib')

# 4. Live Single-Sample Inference Smoke Test
loaded_model = joblib.load('production_models/knn_regressor_pipeline.joblib')
sample_payload = pd.DataFrame([test_dict])
predicted_value = loaded_model.predict(sample_payload)[0]
print(f"Predicted Value: {predicted_value:.2f}")
```

---

## 2. Production Latency & Memory Footprint Rules

1. **Model Storage:** The `.joblib` file will store all $X_{\\text{train}}$ rows. For $N \\le 100,000$, file size is manageable ($< 50\\text{ MB}$).
2. **Inference Latency:** A single query with KD-Tree/Ball-Tree takes $< 1\\text{ ms}$, easily satisfying sub-10ms web SLAs.
