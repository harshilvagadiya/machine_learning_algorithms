# Guide 05: Search Algorithms — Brute vs KD-Tree vs Ball-Tree

> **Note on Workflow:**
> For pipeline assembly and inference deployment, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **indexing algorithms under the hood of `KNeighborsClassifier`**.

---

## 1. The Nearest Neighbor Search Problem

When predicting on a single query point $\mathbf{x}^*$, how does scikit-learn find the $K$ closest neighbors out of $N$ training rows?

In `KNeighborsClassifier(algorithm=...)`, there are 4 options:
1. `'brute'` (Brute Force Exhaustive Search)
2. `'kd_tree'` ($k$-dimensional Tree Index)
3. `'ball_tree'` (Metric Ball Tree Index)
4. `'auto'` (Heuristic selection by scikit-learn)

---

## 2. Head-to-Head Comparison

| Search Algorithm | Construction Time | Query Time ($D < 20$) | Query Time ($D > 20$) | Compatible Metrics | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`brute`** | $O(1)$ (No index built) | $O(N \cdot D)$ (Slow for large $N$) | $O(N \cdot D)$ (Predictable) | **All** distance metrics | Smallest (Raw data only) |
| **`kd_tree`** | $O(D \cdot N \log N)$ | $O(D \log N)$ (Lightning fast) | Degrades to $O(N \cdot D)$ | Euclidean ($L_2$), Manhattan ($L_1$), Chebyshev | Moderate (Tree pointers) |
| **`ball_tree`** | $O(D \cdot N \log N)$ | $O(D \log N)$ (Fast) | Outperforms KD-Tree for $D \sim 20-50$ | Any metric obeying triangle inequality (e.g. Haversine) | Moderate (Tree + radii) |

---

## 3. How KD-Tree and Ball-Tree Work

* **KD-Tree:** Splits space along axis-aligned medians alternating through dimensions. Degrades when $D > 20$ because search sphere intersects almost all bounding hyperplanes.
* **Ball-Tree:** Groups points into nested hyperspheres (balls). Uses triangle inequality to prune distant hyperspheres without computing point distances.
* **`leaf_size`:** Controls when tree splitting stops (default 30). Smaller means deeper tree (faster in low-D, more RAM).
