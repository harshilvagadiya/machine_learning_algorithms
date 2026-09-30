# 🎲 Shrinkage, Learning Rate & Stochastic Subsampling in GBM

Stochastic Gradient Boosting (Friedman, 2002) adds randomized subsampling to function-space gradient descent, creating an ensemble that blends the variance-reduction benefits of Bagging with the bias-reduction power of Boosting.

---

## 1. Stochastic Subsampling (`subsample < 1.0`)

At each iteration $m$, instead of using all $N$ training samples:
- Draw a random subsample of size $\lfloor \text{subsample} \times N \rfloor$ without replacement.
- Calculate pseudo-residuals and fit tree $h_m(x)$ on this random subset only.
- Recommended setting: `subsample=0.8` (or $0.6 - 0.85$).

### Benefits:
1. **Variance Reduction:** Independent random subsamples break correlation among successive trees.
2. **Computational Speedup:** Each tree trains on $20\%-40\%$ fewer rows.
3. **Out-of-Bag Internal Validation:** The unselected rows provide an independent internal estimate of test error at every boosting step.

---

## 2. Feature Subsampling (`max_features < 1.0`)

Similar to Random Forest, subsampling candidate features per split or per tree (`max_features='sqrt'` or `0.8`) further decorrelates trees and prevents single dominant features from masking subtle residual patterns.
