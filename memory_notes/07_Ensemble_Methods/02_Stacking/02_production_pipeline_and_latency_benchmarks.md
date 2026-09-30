# 🚀 Stacking Ensembles: Production Deployment & Latency Benchmarks

Stacking delivers state-of-the-art predictive accuracy by learning the optimal weighting of distinct models, but its production deployment demands careful latency budgeting.

---

## 1. Production Inference Flow & Latency Dynamics

During online real-time inference on a new observation $x_{\text{new}}$:
1. $x_{\text{new}}$ passes through the preprocessor.
2. Every Level-0 base estimator independently computes $\hat{y}_m(x_{\text{new}})$ or $P_m(y \mid x_{\text{new}})$.
3. The resulting Level-0 prediction vector $\tilde{x}_{\text{new}}$ (optionally concatenated with $x_{\text{new}}$ if `passthrough=True`) is fed to the Level-1 Meta-Learner.
4. The Meta-Learner computes final prediction $\hat{y}_{\text{final}}$.

$$\tau_{\text{stacking}} = \tau_{\text{preprocessor}} + \sum_{m=1}^{M} \tau_{m,\text{base}} + \tau_{\text{meta}}$$

### Latency Profiles Across Typical Stacks
| Level-0 Base Learners | Meta-Learner | Typical Latency ($\tau$) | SLA Suitability |
| :--- | :--- | :--- | :--- |
| OLS + Ridge + Lasso | ElasticNet | $< 0.20$ ms | High-frequency trading, real-time ad bidding |
| OLS + DecisionTree + KNN | Ridge | $1.20 - 3.50$ ms | E-commerce checkout, loan underwriting |
| Random Forest + XGBoost + LightGBM | Logistic Regression | $2.50 - 8.00$ ms | Fraud scoring, insurance claims triage |

---

## 2. Meta-Learner Selection Best Practices

1. **Keep Level-1 Simple and Regularized:**
   - **Never** use a complex, deep tree ensemble (e.g. Random Forest or XGBoost) as the Level-1 meta-learner. It will severely overfit the meta-features.
   - **Always** use a simple linear model with $L_1$ or $L_2$ shrinkage:
     - For Regression: `RidgeCV(alphas=np.logspace(-3, 3, 20))` or `LassoCV`.
     - For Classification: `LogisticRegression(C=1.0, penalty='l2')`.

2. **Passthrough Consideration:**
   - Setting `passthrough=True` gives the meta-learner direct access to the raw features alongside base model predictions.
   - Recommended when base models specialize in non-linear partitions but a linear residual trend exists in the raw input space.

---

## 3. Production Deployment Code Template

```python
import joblib
import time

# Export serialized stacking pipeline
joblib.dump(enterprise_stack, 'production_models/stacking_pipeline.joblib', compress=3)

# Inference Service Layer
class StackingInferenceService:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)
    
    def predict(self, sample_df):
        t0 = time.perf_counter()
        pred = self.pipeline.predict(sample_df)[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "prediction": float(pred),
            "latency_ms": round(latency_ms, 3),
            "sla_met": latency_ms < 2.0
        }
```
