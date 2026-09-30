# 🌲 Random Forest: Bagging & Bootstrap Aggregation

Random Forest (Breiman, 2001) builds an ensemble of $B$ deep, unpruned decision trees on bootstrap samples drawn with replacement:
$$\hat{y} = \text{mode} \{ h_1(x), \dots, h_B(x) \}$$
Reduces total variance while preserving the low bias of deep trees.
