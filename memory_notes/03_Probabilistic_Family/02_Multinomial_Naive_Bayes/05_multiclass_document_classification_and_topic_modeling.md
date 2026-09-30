# 📚 Multiclass Document Classification & Topic Triage

MultinomialNB natively scales to hundreds of target classes without one-vs-rest overhead.
Because each class has its own vector $p_c \in \mathbb{R}^d$, computing class scores is a single sparse matrix multiplication:
$$\text{Scores} = X \cdot \ln(P^T) + \ln(\text{Prior})$$
Can process $10,000$ documents per second on a single CPU core!
