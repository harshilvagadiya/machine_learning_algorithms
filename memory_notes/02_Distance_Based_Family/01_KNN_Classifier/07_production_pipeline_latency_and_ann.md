# Guide 07: Production Pipeline, Latency SLAs & Modern ANN Migration

> **Note on Workflow:**
> For general model serialization patterns, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **production latency constraints, memory footprint, and Approximate Nearest Neighbor (ANN) alternatives**.

---

## 1. The Production Dilemma: Storage & Latency SLAs

```
Dataset Size N     Scikit-Learn KNN Query Time     Production Viability
─────────────────────────────────────────────────────────────────────────
N <= 50,000        0.1 ms - 2.0 ms                 ✅ Excellent (Fast & Simple)
N = 500,000        15 ms - 80 ms                   ⚠️ Marginal (Requires KD/BallTree)
N >= 5,000,000     500 ms - 3000 ms                ❌ Unusable for Real-Time SLAs
```

### Memory Footprint Warning:
* In a linear model, saving the pipeline with `joblib.dump()` creates a file of **$5\text{ KB} - 50\text{ KB}$**, regardless of whether you trained on 100 rows or 100 million rows.
* In KNN, `joblib.dump()` **serializes the entire training matrix $X_{\text{train}}$**!

---

## 2. When to Migrate to Approximate Nearest Neighbors (ANN)?

When $N > 100,000$ or dimensions $D > 50$ (e.g. text/image embeddings, vector databases):
Exact nearest neighbor search ($100\%$ precision) is replaced with **Approximate Nearest Neighbors ($98-99\%$ recall with $100\times$ speedup)**.

| Framework | Backed By | Core Algorithm | Typical Use Case |
| :--- | :--- | :--- | :--- |
| **HNSWlib** | Open-source | Hierarchical Navigable Small World graphs | General vector search, ultra-fast queries |
| **Faiss** | Meta AI | Inverted File Index (IVF) + Product Quantization (PQ) | Billion-scale GPU/CPU vector search |
| **Annoy** | Spotify | Random projection trees | Read-only memory-mapped recommendations |
| **ScaNN** | Google Research | Anisotropic vector quantization | Ultra-low latency search on Google Cloud |
