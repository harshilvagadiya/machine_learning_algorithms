# ✂️ Pruning & Regularization Hyperparameters

Unconstrained trees will memorize every individual training point ($R^2 = 1.0$, infinite overfitting).
Regularize via:
- `max_depth`: Depth limits ($4 - 8$).
- `min_samples_split`: Minimum node size before splitting ($10 - 50$).
- `min_samples_leaf`: Minimum samples per leaf ($5 - 20$).
- `ccp_alpha`: Minimal Cost-Complexity Post-Pruning parameter.
