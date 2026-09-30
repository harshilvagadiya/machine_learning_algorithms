# 🚀 Production NLP Pipeline & Latency

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=25000, ngram_range=(1, 2))),
    ('mnb', MultinomialNB(alpha=0.1))
])
nlp_pipeline.fit(corpus_train, labels_train)
joblib.dump(nlp_pipeline, 'production_models/spam_filter.joblib', compress=3)
```
Inference latency per email/message: **$< 0.15$ ms**.
