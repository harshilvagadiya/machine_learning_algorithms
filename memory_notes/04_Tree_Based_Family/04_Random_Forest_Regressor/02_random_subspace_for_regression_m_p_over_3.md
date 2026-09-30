# 📐 Random Subspace for Regression ($m = p / 3$)

While classification defaults to $m = \sqrt{p}$, Leo Breiman proved that for continuous regression targets:
$$m = \lfloor p / 3 \rfloor$$
provides the optimal bias-variance tradeoff across regression benchmark problems.
