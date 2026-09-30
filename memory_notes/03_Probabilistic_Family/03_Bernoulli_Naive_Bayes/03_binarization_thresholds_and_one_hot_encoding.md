# ✂️ Binarization Thresholds & Feature Encoding

`BernoulliNB(binarize=0.0)` automatically maps continuous inputs:
$$x_j' = \begin{cases} 1 & \text{if } x_j > \text{binarize} \\ 0 & \text{if } x_j \le \text{binarize} \end{cases}$$
If features are already boolean/binary, set `binarize=None`.
