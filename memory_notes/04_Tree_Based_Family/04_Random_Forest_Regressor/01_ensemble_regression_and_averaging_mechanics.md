# 🌲 Random Forest Regression: Averaging Mechanics

Random Forest for regression averages $B$ unpruned regression trees:
$$\hat{y}(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)$$
Averages out the high variance of individual deep trees while keeping bias low.
