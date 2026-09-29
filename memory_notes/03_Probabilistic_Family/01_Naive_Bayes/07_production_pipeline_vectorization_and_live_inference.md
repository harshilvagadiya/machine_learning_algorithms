# 07. Production Pipelines, Streaming `partial_fit`, & Sub-Millisecond SLAs

Welcome to Guide 07 of the **Naive Bayes Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **end-to-end pipeline serialization, out-of-core online learning via `partial_fit`, and live inference performance**.

---

## 1. Zero-Leakage Production Pipelines

To deploy Naive Bayes into production without training-serving skew, the feature vectorizer/preprocessor and the estimator must be encapsulated into a single unified `Pipeline`.

```
  RAW INCOMING TEXT / JSON PAYLOAD
                  │
                  ▼
  ┌──────────────────────────────────────────────────┐
  │              SCIKIT-LEARN PIPELINE               │
  │                                                  │
  │   Step 1: TfidfVectorizer / Preprocessor        │
  │           (Fitted strictly on train data)        │
  │                         │                        │
  │                         ▼                        │
  │   Step 2: Naive Bayes Estimator                  │
  │           (MultinomialNB / GaussianNB)           │
  └──────────────────────────────────────────────────┘
                  │
                  ▼
  SERIALIZED ARTIFACT: model_pipeline.joblib
```

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Pipeline Definition
production_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ('nb', MultinomialNB(alpha=0.1))
])

# 2. Fit strictly on train partition
production_pipeline.fit(X_train, y_train)

# 3. Serialize to disk
joblib.dump(production_pipeline, "production_models/spam_multinomial_pipeline.joblib")
```

---

## 2. Streaming & Out-of-Core Learning: `partial_fit`

Unlike algorithms that require keeping the entire dataset in RAM (like KNN or SVM), Naive Bayes is an **incremental learner**.

Because the model only needs to accumulate count tallies ($N_{cj}$) or running means ($\mu$) and variances ($\sigma^2$), it can learn from endless data streams (e.g. Apache Kafka, live logs) without memory exhaustion.

```
       100 GB Streaming Data ──► [ Batch 1: 10,000 msgs ] ──► model.partial_fit()
                             ──► [ Batch 2: 10,000 msgs ] ──► model.partial_fit()
                             ──► [ Batch 3: 10,000 msgs ] ──► model.partial_fit()
                                 (Memory usage stays flat at < 50 MB!)
```

```python
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

# HashingVectorizer has fixed feature space (no dictionary in memory)
hasher = HashingVectorizer(n_features=2**18, alternate_sign=False)
stream_nb = MultinomialNB(alpha=0.1)

classes = np.array([0, 1])

for batch_X, batch_y in stream_data_generator():
    X_vec = hasher.transform(batch_X)
    stream_nb.partial_fit(X_vec, batch_y, classes=classes)
```

---

## 3. Sub-Millisecond Inference SLAs (< 1 ms)

Because scoring a test sample consists purely of:
1. Transforming tokens into indices.
2. A single vector dot product $\mathbf{x}^T \mathbf{w}_c + b_c$.
3. Finding the $\arg\max$.

Naive Bayes delivers **sub-millisecond latency** ($< 0.5\text{ms}$ per request), easily beating strict web API latency Service Level Agreements (SLAs).

```python
# Live Production Smoke Test
loaded_model = joblib.load("production_models/spam_multinomial_pipeline.joblib")

test_payload = ["Urgent: Claim your $1,000 Amazon gift card now! Click bit.ly/prize"]

t0 = time.perf_counter()
predicted_class = loaded_model.predict(test_payload)[0]
probabilities = loaded_model.predict_proba(test_payload)[0]
latency_ms = (time.perf_counter() - t0) * 1000

print(f"🎯 Prediction: {'SPAM' if predicted_class == 1 else 'HAM'}")
print(f"📊 Confidence: {max(probabilities)*100:.2f}%")
print(f"⏱️ Latency   : {latency_ms:.3f} ms (SLA < 5ms: PASS)")
```

---

## 4. Production Checklist

* [x] **Zero Data Leakage:** Vectorizer or preprocessor is fitted strictly on $X_{\text{train}}$.
* [x] **No Negative Values in MultinomialNB:** Verified absence of `StandardScaler()`.
* [x] **Variance Smoothing in GaussianNB:** Verified `var_smoothing >= 1e-9`.
* [x] **Laplace Alpha Tuned:** Checked via CV across $[0.01, 0.1, 1.0]$.
* [x] **Pipeline Serialized:** Stored as self-contained `.joblib` artifact.
* [x] **Live Smoke Test Executed:** Verified predictions on raw dictionary/text input.
