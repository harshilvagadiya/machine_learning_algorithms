# ⚖️ The Absence Penalty: Bernoulli vs. Multinomial

The critical mathematical difference:
- **MultinomialNB:** Ignores words not present in the document ($x_j = 0$ contributes $p_{jc}^0 = 1$).
- **BernoulliNB:** Actively penalizes the **absence** of expected words!
  $$x_j = 0 \implies (1 - p_{jc})^1$$
If a medical document is missing the word "fever", BernoulliNB explicitly penalizes diseases where "fever" has high prevalence $p_{jc} \approx 0.99$!
