# 🎛️ Tuning Alpha in Natural Language Processing

In text classification with sparse vocabulary ($d > 50,000$):
- $\alpha = 1.0$: Default Laplace smoothing. Can over-smooth rare domain-specific keywords.
- $\alpha = 0.1$: Often optimal for large text corpora (e.g. 20 Newsgroups, spam detection).
- Grid search over $\alpha \in [0.01, 0.05, 0.1, 0.5, 1.0]$ using StratifiedKFold.
