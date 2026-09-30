# 🚫 The Critical Rule: NEVER Apply StandardScaler to Count/TF-IDF Matrices

StandardScaler subtracts the feature mean $\mu_j$:
$$x_{ij}' = x_{ij} - \mu_j$$
This produces **negative values** and **destroys sparse CSR matrix storage** (converting 99% zeros into dense floating-point values and overflowing RAM).
Multinomial Naive Bayes requires non-negative counts $x_{ij} \ge 0$.
**Rule:** Use `MaxAbsScaler` (preserves sparsity and non-negativity) or feed raw `CountVectorizer` / `TfidfTransformer` directly to MultinomialNB!
