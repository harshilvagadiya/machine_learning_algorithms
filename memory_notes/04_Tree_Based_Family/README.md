# 🌳 04. Tree-Based Family Master Protocol & Step Standard

> **Dumb Student Summary (Dimag mein hamesha ke liye fit karne wali baat):**  
> Chahe aap **Decision Tree** chala rahe hon ya **Random Forest**, chahe **Classification** ho ya **Regression**:  
> **Tree-Based Family ka har project hamesha exact in 13 Unified Steps mein chalta hai!**  
> Is family ka sabse bada superpower hai: **Scale Invariance** (Kisi bhi number ko scale karne ki zaroorat nahi hai)!

---

## 🧭 The Unified 13-Step Standard for Tree-Based Models

```text
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DATA FOUNDATION & HYGIENE (STEPS 1 TO 6)                     │
├────────────────────────────────────────────────────────────────────────┤
│  STEP 1: Universal Libraries Import                                    │
│  STEP 2: Data Ingestion & 3-Sec Audit                                  │
│  STEP 2B: String Sanitization (Clean $, %, strings to numeric)         │
│  STEP 3: 3-Bucket Feature Segregator (Continuous, Categorical, Drop)    │
│  STEP 4: Data Hygiene (Duplicates & Missing Values Audit)              │
│  STEP 5: Train-Test Split (Zero-Leakage Protocol, Stratified if Clf)    │
│  STEP 6: Target Profiler (Skewness for Reg / Class Balance for Clf)    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: TREE PREPROCESSING & MODELING (STEPS 7 TO 13)                │
├────────────────────────────────────────────────────────────────────────┤
│  STEP 7: Scale-Invariant Tree Preprocessor                             │
│          • Median Imputer for Numbers (NO StandardScaler!)             │
│          • Missing Imputer + OneHotEncoder for Categorical             │
│          • Crash-Proof `if len > 0:` Dynamic Guards                    │
│                                                                        │
│  STEP 8: Baseline Reality Check / Overfitting Proof                    │
│          • Decision Trees : Unconstrained Tree (max_depth=None)        │
│          • Random Forests : Baseline Single Tree vs Forest             │
│                                                                        │
│  STEP 9: Hyperparameter Optimization (GridSearchCV with 5-Fold CV)     │
│          • Depth (max_depth) & Leaf size (min_samples_leaf)            │
│          • Forest size (n_estimators) & Subspace (max_features)       │
│                                                                        │
│  STEP 10: Model Benchmark Duel (Tree/Forest vs Baseline)               │
│          • Regression     : Tree vs Ridge Regression                   │
│          • Classification : Tree vs Logistic Regression / KNN          │
│                                                                        │
│  STEP 11: Feature Importance & Tree Diagnostics                        │
│          • Gini / Variance Impurity Importance (MDI)                   │
│          • Residual Analysis / Confusion Matrix & ROC Curve            │
│                                                                        │
│  STEP 12: Production Pipeline Assembly & .joblib Serialization         │
│  STEP 13: Sub-50ms Inference Latency Smoke Test (assert avg < 50ms)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 The Unified Code Blueprint: Steps 7 Through 13

### ⚙️ STEP 7: Crash-Proof Tree Preprocessor (No StandardScaler!)
```python
# ==============================================================================
# ⚙️ STEP 7: TREE-OPTIMIZED UNIVERSAL PREPROCESSOR (SCALE-INVARIANT)
# ==============================================================================
def build_tree_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    transformers = []
    
    # 1. Continuous / Numerical (Imputer only, NO SCALING for Trees!)
    if len(num_cols) > 0:
        transformers.append(('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols))
        
    # 2. Categorical / Text (Mode Imputer + OHE)
    if len(cat_cols) > 0:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))
        
    return ColumnTransformer(transformers=transformers, remainder='drop') if len(transformers) > 0 else 'passthrough'

preprocessor = build_tree_preprocessor(X_train)
print("✅ STEP 7: Scale-Invariant Tree Preprocessor Ready.")
```

---

### 🚨 STEP 8: Baseline Reality Check (Overfitting Demonstration)
```python
# ==============================================================================
# 🚨 STEP 8: UNCONSTRAINED BASELINE FIT (OVERFITTING PROOF)
# ==============================================================================
# Unconstrained tree memorizes training noise
uncon_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=None, random_state=42) if not is_classification 
              else DecisionTreeClassifier(max_depth=None, random_state=42))
])
uncon_pipe.fit(X_train, y_train)

tr_score = uncon_pipe.score(X_train, y_train)
te_score = uncon_pipe.score(X_test, y_test)
print(f"Unconstrained Train Score : {tr_score*100:.2f}% (Pure Memorization)")
print(f"Unconstrained Test Score  : {te_score*100:.2f}% (Variance Collapse)")
print(f"Generalization Gap Drop   : {(tr_score - te_score)*100:.2f}% Drop!")
```

---

### 🎯 STEP 9: Hyperparameter Optimization (GridSearchCV via 5-Fold CV)
```python
# ==============================================================================
# 🎯 STEP 9: HYPERPARAMETER TUNING VIA 5-FOLD CROSS-VALIDATION
# ==============================================================================
if not is_classification:
    param_grid = {
        'model__max_depth': [4, 6, 8, 10, 12],
        'model__min_samples_split': [5, 10, 20],
        'model__min_samples_leaf': [2, 5, 10],
        'model__criterion': ['squared_error', 'absolute_error']
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'r2'
else:
    param_grid = {
        'model__max_depth': [3, 5, 7, 9],
        'model__min_samples_leaf': [2, 5, 10],
        'model__criterion': ['gini', 'entropy']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'accuracy'

full_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', DecisionTreeRegressor(random_state=42) if not is_classification 
              else DecisionTreeClassifier(random_state=42))
])

grid = GridSearchCV(full_pipeline, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
grid.fit(X_train, y_train)

best_tree = grid.best_estimator_
print(f"🏆 Best Hyperparameters : {grid.best_params_}")
print(f"🎯 Best 5-Fold CV Score  : {grid.best_score_*100:.2f}%")
```

---

### ⚔️ STEP 10: Model Benchmark Duel (Tree vs Baseline Model)
```python
# ==============================================================================
# ⚔️ STEP 10: ALGORITHM DUEL (TREE VS LINEAR/BASELINE SHOWDOWN)
# ==============================================================================
from sklearn.linear_model import Ridge, LogisticRegression

if not is_classification:
    baseline_pipe = Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler(with_mean=False)), # Linear models require scaling
        ('linear', Ridge(alpha=1.0, random_state=42))
    ])
else:
    baseline_pipe = Pipeline([
        ('prep', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('linear', LogisticRegression(max_iter=1000, random_state=42))
    ])

baseline_pipe.fit(X_train, y_train)

tree_test_score = best_tree.score(X_test, y_test)
baseline_test_score = baseline_pipe.score(X_test, y_test)

print("=" * 65)
print(f"Tuned Tree Model Test Score     : {tree_test_score*100:.2f}%")
print(f"Baseline Linear Model Test Score: {baseline_test_score*100:.2f}%")
print(f"Tree Improvement Over Baseline  : {(tree_test_score - baseline_test_score)*100:+.2f}%")
print("=" * 65)
```

---

### 📊 STEP 11: Feature Importance & Error Diagnostics
```python
# ==============================================================================
# 📊 STEP 11: FEATURE IMPORTANCE & ERROR DIAGNOSTICS DASHBOARD
# ==============================================================================
fitted_model = best_tree.named_steps['model']
importances = fitted_model.feature_importances_

# Extract feature names safely from preprocessor
try:
    feature_names = best_tree.named_steps['prep'].get_feature_names_out()
except:
    feature_names = [f"feat_{i}" for i in range(len(importances))]

imp_df = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 4.5))
imp_df.plot(kind='barh', color='forestgreen', edgecolor='black')
plt.title("Top 10 Feature Importances (MDI Gini/Variance Reduction)", fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

---

### 🏭 STEP 12 & 13: Serialization & Latency Smoke Test
```python
# ==============================================================================
# 🏭 STEP 12 & 13: PRODUCTION SERIALIZATION & LATENCY SMOKE TEST
# ==============================================================================
import time

save_path = Path("production_models/tree_model_pipeline.joblib")
save_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_tree, save_path)
print(f"💾 Model Serialized to: {save_path.resolve()}")

# Single-sample sub-50ms latency verification
sample = X_test.iloc[[0]]
t0 = time.time()
for _ in range(100):
    best_tree.predict(sample)
avg_lat_ms = ((time.time() - t0) / 100) * 1000

print(f"⚡ Single-Sample Serving Latency: {avg_lat_ms:.3f} ms")
assert avg_lat_ms < 50.0, "Latency SLA breach (> 50ms)!"
print("✅ Production Readiness Confirmed: Sub-50ms SLA Passed.")
```

---

## 🧭 Sub-Family Differences Cheat Sheet

| Step | Decision Tree Classifier | Decision Tree Regressor | Random Forest Classifier | Random Forest Regressor |
| :--- | :--- | :--- | :--- | :--- |
| **Step 7 (Scale)** | Imputer + OHE (No Scale) | Imputer + OHE (No Scale) | Imputer + OHE (No Scale) | Imputer + OHE (No Scale) |
| **Step 8 (Reality Check)**| Unconstrained (100% Acc) | Unconstrained (100% R²) | Single Tree vs Forest | Single Tree vs Forest |
| **Step 9 (Criterion)** | `gini`, `entropy` | `squared_error`, `absolute_error` | `gini`, `entropy` | `squared_error`, `absolute_error` |
| **Step 10 (Duel)** | Tree vs Logistic / KNN | Tree vs Ridge Regression | Forest vs Single Tree | Forest vs Single Tree |
| **Step 11 (Plot)** | Confusion Matrix + MDI | Residual Plot + MDI | OOB Score + MDI | Extrapolation Plot + MDI |
| **Step 12/13 (Deploy)**| `.joblib` (< 0.1ms SLA) | `.joblib` (< 0.1ms SLA) | `.joblib` (< 2.0ms SLA) | `.joblib` (< 2.0ms SLA) |
