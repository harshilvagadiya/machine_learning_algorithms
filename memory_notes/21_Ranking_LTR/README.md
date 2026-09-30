# 🎯 Learning-to-Rank (LTR / LambdaMART) Architecture & Memory Notes

## 1. Executive Summary & Paradigm Placement
Learning-to-Rank (LTR) algorithms optimize the **relative ordering of candidate items within specific query groups or sessions $\mathcal{Q}$** (e.g. search engine ranking, recommender feed prioritization, candidate resume screening, loan allocation).
Unlike standard regression or classification where each sample is evaluated independently, in LTR the prediction quality is evaluated **collectively across the entire permutation list**.

---

## 2. The Three Generations of LTR

```
Pointwise LTR (1st Gen)       Pairwise LTR (2nd Gen)        Listwise / LambdaMART (3rd Gen)
      [ Item 1 ] -> y1             (Item 1 > Item 2)?           [ Item 1, Item 2, Item 3 ]
      [ Item 2 ] -> y2             (Item 2 > Item 3)?                       │
      [ Item 3 ] -> y3                     │                                ▼
             │                             ▼                    Direct NDCG List Permutation
             ▼                     RankNet / Pair Loss             Optimization (LambdaMART)
      Independent MSE
```

| Paradigm | Architecture | Loss Function | Fundamental Limitation |
| :--- | :--- | :--- | :--- |
| **Pointwise** | Standard Regressor / Classifier | $\sum_i (y_i - f(\mathbf{x}_i))^2$ | Ignores group structure; penalties at rank 1 vs rank 50 are identical |
| **Pairwise** | RankNet / Pairwise SVM | Cross-Entropy over relative order $P(i \succ j)$ | Treats all pairs equally; swapping rank 1 and 2 has same loss as swapping 99 and 100 |
| **Listwise** | **LambdaMART (LightGBM)** | **Direct non-differentiable NDCG optimization** | State-of-the-art; requires query group partition metadata |

---

## 3. Mathematical Foundations: LambdaMART & NDCG Optimization

### Normalized Discounted Cumulative Gain (NDCG@K):
$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{y_{(i)}} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
where $\text{IDCG}@K$ is the ideal maximum DCG under sorted ground truth relevance.

### The Lambda Gradient ($\lambda$-Gradient):
NDCG is flat or discontinuous with respect to model weights $\mathbf{w}$, preventing standard gradient descent.
Burges et al. (2010) resolved this in **LambdaMART** by defining virtual gradient forces between items $i$ and $j$:
$$\lambda_{ij} = \frac{-\sigma}{1 + \exp(\sigma(s_i - s_j))} \cdot |\Delta \text{NDCG}_{ij}|$$
where $|\Delta \text{NDCG}_{ij}|$ is the exact change in NDCG achieved by swapping item $i$ and item $j$ in the ranked output list.
The total gradient driving tree leaf updates is:
$$\lambda_i = \sum_{j: (i, j) \in \mathcal{P}} \lambda_{ij} - \sum_{k: (k, i) \in \mathcal{P}} \lambda_{ki}$$

---

## 4. Production Engineering & Serving Latency
- **Query Group Partitioning (`group` parameter):** In production LightGBM ranking pipelines, training data MUST be sorted by query group ID, with an accompanying array of group sizes (e.g. `[15, 22, 18, 40]`).
- **Serving SLA:** At inference time, scoring candidate items within a query requires only evaluating the LightGBM boosted trees on feature matrix $\mathbf{X}_{\text{candidates}}$, followed by an $\mathcal{O}(K \log K)$ sort:
  $$\text{Scoring Latency per Candidate Item} \le 0.02 \text{ ms}$$
  $$\text{Full Query Re-Ranking (100 candidates)} \le 1.8 \text{ ms (P99 < 3.5 ms)}$$
