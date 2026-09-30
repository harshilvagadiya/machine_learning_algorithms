# 📊 Histogram Binning & LightGBM Roots in HistGradientBoosting

Standard Gradient Boosting scales poorly to large datasets ($N > 50,000$) because evaluating continuous split points requires sorting $N$ values for each feature at every node: $\mathcal{O}(D \cdot N \log N)$.
Inspired by Microsoft's LightGBM, scikit-learn introduced `HistGradientBoostingClassifier` and `HistGradientBoostingRegressor` to eliminate continuous sorting.

---

## 1. How Histogram Binning Works

1. **Pre-Binning into 256 Integer Bins:**
   Before tree building begins, continuous feature values are quantized into integer bins (by default, $\le 256$ bins, fitting into a single `uint8` byte in RAM):
   $$x_{ij} \in \mathbb{R} \longrightarrow b_{ij} \in \{0, 1, \dots, 255\}$$

2. **Histogram Construction at Internal Nodes:**
   Instead of scanning sorted values, the algorithm constructs a 256-bucket histogram accumulating the first and second gradients (sum of gradients $G_k$ and sum of hessians $H_k$) for each bin $k \in \{0, \dots, 255\}$:
   $$\mathcal{O}(N) \text{ operations instead of } \mathcal{O}(N \log N)!$$

3. **Histogram Subtraction Trick:**
   When a parent node is split into left and right children, the algorithm computes the histogram for the smaller child node and calculates the larger child's histogram by simple element-wise subtraction:
   $$\text{Hist}_{\text{Right}} = \text{Hist}_{\text{Parent}} - \text{Hist}_{\text{Left}}$$
   This halves the tree construction cost at every depth level!

---

## 2. Native Missing Value Support

In `HistGradientBoosting`:
- You **do not need to impute missing values** (`NaN`).
- During training, missing values are assigned to a dedicated 256th bin.
- At each split, the algorithm evaluates sending missing values to the left child vs. the right child, and automatically picks the direction that maximizes gain!
- Native support for categorical features (`categorical_features=...`) without one-hot explosion!
