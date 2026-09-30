# 🔄 Non-Normal Distributions & Power Transformers

If features exhibit heavy skewness, log-normal tails, or bimodal distributions, standard Gaussian Naive Bayes underperforms.
Solution: Prepend `PowerTransformer(method='yeo-johnson')` or `QuantileTransformer(output_distribution='normal')` to map arbitrary empirical distributions into bell-shaped Gaussian distributions before fitting GNB!

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB

gnb_pipe = Pipeline([
    ('yeo_johnson', PowerTransformer(method='yeo-johnson')),
    ('gnb', GaussianNB())
])
gnb_pipe.fit(X_train, y_train)
```
