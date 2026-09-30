# 📊 Leaf Node Estimation & Step Functions

In each leaf $R_j$, the optimal prediction minimizing mean squared error is the sample mean:
$$\hat{y}_j = \frac{1}{|R_j|} \sum_{x_i \in R_j} y_i$$
The global function is a piecewise constant step function:
$$\hat{y}(x) = \sum_{j=1}^J \hat{y}_j \mathbb{I}(x \in R_j)$$
