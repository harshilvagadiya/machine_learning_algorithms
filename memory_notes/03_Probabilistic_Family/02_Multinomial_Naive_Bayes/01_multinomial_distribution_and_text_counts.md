# 📜 Multinomial Distribution & Text Word Counts

Multinomial Naive Bayes models discrete occurrence counts (e.g. word frequencies in text classification, bag-of-words histograms):

$$P(X = x \mid Y = c) = \frac{(\sum_j x_j)!}{\prod_j x_j!} \prod_{j=1}^d p_{jc}^{x_j}$$

Where $p_{jc}$ represents the probability of term $j$ appearing in a document of class $c$:
$$p_{jc} = \frac{N_{jc} + \alpha}{N_c + \alpha d}$$
