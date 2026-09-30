# 🚀 Gradient Boosting Classifier: Production Latency & Deployment

Gradient Boosting is widely considered the gold standard for structured tabular classification in industry.

---

## 1. Latency Profiles

- To classify an observation, $M$ trees of depth $d \approx 4$ are traversed sequentially.
- With $M = 100$ trees, single-sample CPU latency is typically **$0.25 - 0.70$ ms**.
- Well within strict financial trading and fraud screening SLAs ($< 2.0$ ms).

---

## 2. Production Code Template

```python
import joblib
import time
import pandas as pd

joblib.dump(full_pipeline, 'production_models/gradient_boosting_clf_pipeline.joblib', compress=3)

# Inference Endpoint
model = joblib.load('production_models/gradient_boosting_clf_pipeline.joblib')

def score_fraud_risk(transaction: dict):
    df_sample = pd.DataFrame([transaction])
    t0 = time.perf_counter()
    prob = model.predict_proba(df_sample)[0, 1]
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "fraud_risk_score": round(float(prob), 4),
        "alert_triggered": bool(prob > 0.5),
        "latency_ms": round(latency_ms, 3),
        "sla_met": latency_ms < 2.0
    }
```
