# 🔔 The Gaussian Normal Distribution Assumption

Gaussian Naive Bayes assumes continuous features follow a 1D normal (bell-curve) distribution within each class:

$$P(X_j = x_j \mid Y = c) = \frac{1}{\sqrt{2\pi \sigma_{jc}^2}} \exp\left( -\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right)$$

Parameters estimated from data:
- $\mu_{jc}$: Sample mean of feature $j$ among training samples of class $c$.
- $\sigma_{jc}^2$: Sample variance of feature $j$ among training samples of class $c$.

Log-likelihood decision rule:
$$\hat{y} = \arg\max_c \left[ \ln P(Y = c) - \frac{1}{2} \sum_{j=1}^d \ln(2\pi \sigma_{jc}^2) - \sum_{j=1}^d \frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2} \right]$$
