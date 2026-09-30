# 🚀 Gaussian Naive Bayes: Production Pipeline & Latency

Gaussian Naive Bayes is one of the fastest classifiers in existence:
- Training: $\mathcal{O}(N \cdot D)$ (single pass to compute mean and variance).
- Prediction: $\mathcal{O}(C \cdot D)$ scalar operations per sample.
- Single-sample inference latency is typically **$0.01 - 0.05$ ms**, ideal for high-throughput packet filtering and microsecond alerting.
