# ⚠️ Conditional Independence Violations & Multicollinearity

When two features $X_1$ and $X_2$ are strongly collinear (e.g. correlation $r > 0.9$):
- Naive Bayes counts the shared evidence **twice** in the product $\prod P(X_j \mid Y)$.
- This pushes posterior probabilities unrealistically close to 0.0 or 1.0 (overconfidence).
- **Rule of Thumb:** Remove redundant collinear features or use PCA dimensionality reduction before applying Naive Bayes.
