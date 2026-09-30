# 📦 Out-Of-Bag (OOB) Score & Validation

Roughly $36.8\%$ of training observations are omitted from each bootstrap tree.
Evaluating each sample on the trees where it was out-of-bag yields an unbiased internal validation score (`oob_score=True`) without needing an external cross-validation hold-out set.
