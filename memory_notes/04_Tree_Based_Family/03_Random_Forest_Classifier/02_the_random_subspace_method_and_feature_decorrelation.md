# 🎲 The Random Subspace Method & Feature Decorrelation

At every candidate split node, instead of considering all $p$ features, Random Forest randomly selects a subset of size:
$$m = \lfloor \sqrt{p} \rfloor$$
This prevents dominant features from being chosen at the root of every tree, dramatically decreasing pairwise tree correlation $\rho$ and lowering ensemble variance $\rho \sigma^2$!
