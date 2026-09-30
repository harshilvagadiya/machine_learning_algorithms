# 🚀 AdaBoost: Production Deployment & Latency Benchmarks

AdaBoost built from decision stumps is one of the fastest decision-making tree models available for production inference.

---

## 1. Real-Time Inference Latency

Because each decision stump contains exactly one comparison (`if x[j] <= threshold: return left else right`), evaluating an AdaBoost model with $M = 100$ stumps requires exactly 100 scalar threshold comparisons and 100 scalar additions:

$$\tau_{\text{inference}} = 100 \times \mathcal{O}(1) \approx 0.05 - 0.20 \text{ ms}$$

- Sub-millisecond latency SLA ($< 1.0$ ms) is trivially met on any standard CPU.

---

## 2. Serialization & Serving Example

```python
import joblib
import time
import pandas as pd

# Export pipeline
joblib.dump(full_pipeline, 'production_models/adaboost_pipeline.joblib', compress=3)

# Load and score
model = joblib.load('production_models/adaboost_pipeline.joblib')

def predict_realtime(data_payload: dict):
    df_sample = pd.DataFrame([data_payload])
    t0 = time.perf_counter()
    proba = model.predict_proba(df_sample)[0, 1]
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "churn_probability": round(float(proba), 4),
        "latency_ms": round(latency_ms, 3),
        "sla_met": latency_ms < 2.0
    }
```
