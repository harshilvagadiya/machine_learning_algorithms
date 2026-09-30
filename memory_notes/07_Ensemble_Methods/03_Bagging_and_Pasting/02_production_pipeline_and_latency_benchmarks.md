# 🚀 Bagging Ensembles: Production Deployment & Latency Benchmarks

Deploying Bagging Ensembles requires balancing the number of base estimators against memory footprint and real-time CPU thread contention.

---

## 1. Computational Complexity & Latency

### Training Complexity
- Standard Bagging trains $B$ independent base estimators on $N$ samples with $D$ features.
- If base estimators are unpruned Decision Trees, fitting cost is:
  $$\mathcal{O}(B \cdot D \cdot N \log N)$$
- Because trees are completely independent, training scales linearly with available CPU cores ($n\_jobs=-1$).

### Inference Complexity & Latency
- To predict on a single sample, all $B$ trees must be traversed:
  $$\tau_{\text{inference}} = \mathcal{O}(B \cdot \text{max\_depth})$$
- For $B = 100$ trees of depth $10-15$, single-sample inference typically takes **$0.40 - 1.50$ ms**, easily satisfying enterprise sub-2ms real-time SLAs!

---

## 2. Production Optimization Guidelines

1. **Avoid Heterogeneous Base Learners with Huge Memory Footprints:**
   - Do not wrap `KNeighborsClassifier` in Bagging with large $B$ unless the dataset is small; each tree would store its own reference copy of the data.
2. **Compress Serialized Artifacts:**
   - Bagging models with unpruned trees can grow to $50-200$ MB.
   - Always use `joblib.dump(model, 'model.joblib', compress=3)` to reduce disk IO and load latency by up to $70\%$.
3. **Out-of-Bag (OOB) Free Validation:**
   - In production retraining pipelines, set `oob_score=True` to eliminate the overhead and data splitting cost of outer cross-validation.

---

## 3. Production Deployment Code Template

```python
import joblib
import time
import pandas as pd

# Save compressed pipeline
joblib.dump(enterprise_pipeline, 'production_models/bagging_pipeline.joblib', compress=3)

# Inference Serving Function
model = joblib.load('production_models/bagging_pipeline.joblib')

def score_transaction(features_dict: dict):
    df_single = pd.DataFrame([features_dict])
    t0 = time.perf_counter()
    pred = model.predict(df_single)[0]
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "prediction": pred,
        "latency_ms": round(latency_ms, 3),
        "sla_pass": latency_ms < 2.0
    }
```
