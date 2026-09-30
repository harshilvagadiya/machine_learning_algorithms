# 📉 Regression Trees & Variance Reduction

Decision Tree Regressors partition continuous feature space $\mathbb{R}^d$ into disjoint hyper-rectangles $R_1, \dots, R_J$ by greedily maximizing **Variance Reduction**:

$$\Delta \text{Var}(S, j, s) = \text{Var}(S) - \frac{N_L}{N} \text{Var}(S_L) - \frac{N_R}{N} \text{Var}(S_R)$$
