# 🔲 Bernoulli Distribution & Binary Features

Bernoulli Naive Bayes models multivariate binary features $x_j \in \{0, 1\}$:

$$P(X_j = x_j \mid Y = c) = p_{jc}^{x_j} (1 - p_{jc})^{(1 - x_j)}$$

Where $p_{jc}$ is the probability that indicator feature $j$ is active in class $c$.
Ideal for bag-of-words presence/absence, symptom checklists, and cybersecurity permission bitmasks.
