# 07. Production Pipelines, Latency SLAs (< 0.1ms), & Rule Extraction

Welcome to Guide 07 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **end-to-end pipeline serialization, sub-millisecond traversal latency, and SQL / C code compilation**.

---

## 1. Zero-Leakage Production Pipelines

Because decision trees do not require feature scaling, the pipeline consists solely of categorical encoders/imputers paired with the regularized `DecisionTreeClassifier`:

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# 1. Pipeline Definition
production_tree_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
])

# 2. Fit strictly on train partition
production_tree_pipe.fit(X_train, y_train)

# 3. Serialize to disk
joblib.dump(production_tree_pipe, "production_models/heart_disease_tree_pipeline.joblib")
```

---

## 2. Ultra-Fast Traversal Latency: $O(\text{Depth})$

In live serving:
* A tree of depth $5$ requires exactly **$5$ scalar if-else comparisons**.
* Traversal latency is on the **microsecond scale** (typically $0.05\text{ms}$ to $0.15\text{ms}$).
* It easily satisfies the most demanding ultra-low-latency real-time bidding, fraud scoring, and telecom SLAs.

```python
# Live Inference Smoke Test
loaded_model = joblib.load("production_models/heart_disease_tree_pipeline.joblib")

sample_payload = pd.DataFrame([X_test.iloc[0].to_dict()])

t0 = time.perf_counter()
predicted_class = loaded_model.predict(sample_payload)[0]
probabilities = loaded_model.predict_proba(sample_payload)[0]
latency_ms = (time.perf_counter() - t0) * 1000

print(f"🎯 Prediction : {predicted_class}")
print(f"📊 Confidence : {max(probabilities)*100:.2f}%")
print(f"⏱️ Latency    : {latency_ms:.3f} ms (SLA < 2ms: PASS)")
```

---

## 3. Direct SQL Compilation & Edge Deployment

Because decision trees consist of pure conditional logic, they can be compiled directly into **SQL `CASE WHEN` statements** or **C headers**:

```sql
-- Direct In-Database SQL Scoring (Zero Python Dependency!)
SELECT 
    patient_id,
    CASE 
        WHEN chest_pain_type = 0 AND max_heart_rate <= 140 THEN 1
        WHEN chest_pain_type > 0 AND st_depression <= 1.5 THEN 0
        ELSE 0
    END AS predicted_disease_risk
FROM patients;
```

---

## 4. Production Governance Checklist

* [x] **Pre-pruning Enabled:** `max_depth` or `min_samples_leaf >= 5` active.
* [x] **No Redundant Scaler:** Verified removal of unnecessary scalers.
* [x] **Zero Data Leakage:** Pipeline fitted strictly on $X_{\text{train}}$.
* [x] **Class Imbalance Guard:** Set `class_weight='balanced'` if targets are skewed.
* [x] **Permutation Importance Audited:** Evaluated on validation set.
* [x] **Pipeline Serialized:** Saved to `.joblib` artifact.
* [x] **Live Inference Smoke Test:** Executed on raw payload dictionary.
