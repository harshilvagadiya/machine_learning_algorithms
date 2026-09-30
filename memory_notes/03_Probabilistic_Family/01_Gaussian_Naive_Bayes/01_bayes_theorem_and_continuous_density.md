# 🎲 Bayes' Theorem & Continuous Density in Gaussian Naive Bayes

Bayes' Theorem provides the mathematical foundation for probabilistic classification:

$$P(Y = c \mid X = x) = \frac{P(Y = c) \cdot P(X = x \mid Y = c)}{P(X = x)}$$

Where:
- $P(Y = c)$ is the **Prior Probability** of class $c$.
- $P(X = x \mid Y = c)$ is the **Class-Conditional Likelihood**.
- $P(X = x) = \sum_k P(Y = k) P(X = x \mid Y = k)$ is the **Evidence** (constant normalizing factor).
- $P(Y = c \mid X = x)$ is the **Posterior Probability**.

The "Naive" assumption asserts that features $X_1, X_2, \dots, X_d$ are mutually conditionally independent given the class:
$$P(X_1, \dots, X_d \mid Y = c) = \prod_{j=1}^d P(X_j \mid Y = c)$$
