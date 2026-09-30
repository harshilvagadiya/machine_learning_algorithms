# ⚠️ The Extrapolation Limit in Tree Ensembles

> [!WARNING]
> Tree-based models (Decision Trees, Random Forests, Extra Trees, Gradient Boosting) **CANNOT extrapolate** outside the minimum and maximum target bounds seen during training!
> If test inputs have values higher than any seen in training, the tree returns the mean of the highest leaf: $\hat{y} = y_{\max}$.
> For time-series trends or unbounded growth, use Linear/Spline models or detrend data first!
