# 🔄 Scale Invariance & Feature Interactions

Decision trees are completely invariant to monotonic transformations of features:
- Scaling or log-transforming features does not change split ordering!
- Tree paths ($X_1 \le 5 \to X_2 > 10$) naturally capture high-order non-linear feature interactions without manual interaction terms.
