# 06. Bernoulli & Complement Naive Bayes: Binary Features & Class Imbalance

Welcome to Guide 06 of the **Naive Bayes Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **the Bernoulli binary formulation, the penalty of absence, and Complement Naive Bayes for severe class imbalances**.

---

## 1. Bernoulli Naive Bayes (`BernoulliNB`)

While `MultinomialNB` tracks *how many times* a word appears ($0, 1, 2, \dots$), **Bernoulli Naive Bayes** tracks strictly *whether* a feature occurs or not ($x_j \in \{0, 1\}$).

The conditional probability mass function follows a product of independent Bernoulli trials:

$$P(\mathbf{x} \mid y = c) = \prod_{j=1}^{D} p_{cj}^{x_j} (1 - p_{cj})^{(1 - x_j)}$$

In log-space:

$$\log P(\mathbf{x} \mid y = c) = \sum_{j=1}^D \left[ x_j \ln(p_{cj}) + (1 - x_j) \ln(1 - p_{cj}) \right] + \ln P(y = c)$$

```
                               BERNOULLI'S DUAL SIGNAL
                               
   Word is Present (x_j = 1):  Adds  ln( p_cj )         [Presence Evidence]
   Word is Absent  (x_j = 0):  Adds  ln( 1 - p_cj )     [Absence Evidence!]
```

### The Critical Architectural Difference: The Penalty of Absence
* In **MultinomialNB**, if a word is absent from a test document ($x_j = 0$), it contributes $0 \times \ln(p_{cj}) = 0$. It does **not** penalize the class.
* In **BernoulliNB**, if a word is absent ($x_j = 0$), it contributes $(1 - 0) \times \ln(1 - p_{cj})$. If that word is normally expected in $90\%$ of spam messages (e.g. *"http"*, *"click"*), its absence actively **dampens the posterior probability of spam**!
* **Best Fit:** Short texts (tweets, SMS, subject lines), binary customer feature flags, medical symptom present/absent checklists.

---

## 2. Complement Naive Bayes (`ComplementNB`): Conquering Class Imbalance

Standard `MultinomialNB` suffers from a well-documented bias when training on severely imbalanced datasets (e.g., $95\%$ normal transactions, $5\%$ fraud).

```
   MAJORITY CLASS (95% of data)  ---> Parameters θ_c receive massive sample counts,
                                      over-concentrating parameter weights.
   MINORITY CLASS (5% of data)   ---> Parameters θ_c receive sparse counts,
                                      causing severe under-prediction of minority class.
```

### The Complement Formulation (Rennie et al., 2003)
Instead of asking *"How likely is word $j$ in class $c$?"*, **Complement Naive Bayes** estimates the likelihood that word $j$ belongs to **all other classes EXCEPT class $c$**:

$$\hat{\theta}_{c, j} = \frac{N_{\neg c, j} + \alpha}{N_{\neg c} + \alpha D}$$

Where:
* $N_{\neg c, j} = \sum_{k \ne c} N_{k, j}$: Count of word $j$ in all training samples that do *not* belong to class $c$.
* $N_{\neg c} = \sum_{k \ne c} N_k$: Total word count across all non-$c$ classes.

By learning from the complement of each class, `ComplementNB` constructs weights that are **unbiased by class size disparities**:

$$w_{cj} = -\ln \hat{\theta}_{cj}$$
$$\hat{y} = \arg\min_{c} \sum_{j=1}^D x_j w_{cj} \quad \text{(or length-normalized)}$$

> **Benchmark Verdict:** On imbalanced text datasets (legal discovery, medical notes, spam with low spam ratio), `ComplementNB` consistently beats standard `MultinomialNB` by **$3\%$ to $8\%$ macro F1-score**.

---

## 3. Master Naive Bayes Variant Selection Matrix

Use this definitive architectural reference when selecting your estimator:

```
  ┌─────────────────┬───────────────────────────────┬────────────────────────────────────────┐
  │ ESTIMATOR       │ INPUT DATA DOMAIN             │ CANONICAL PRODUCTION USE CASE          │
  ├─────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ GaussianNB      │ Continuous real numbers       │ Medical diagnostics, sensor telemetry, │
  │                 │ (-inf to +inf, bell-shaped)   │ low-latency embedded systems.          │
  ├─────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ MultinomialNB   │ Non-negative counts / TF-IDF  │ Text classification, document topics,  │
  │                 │ (x_j >= 0, balanced classes)  │ SMS spam detection with balanced data. │
  ├─────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ BernoulliNB     │ Binary boolean flags          │ Tweet sentiment, short SMS queries,    │
  │                 │ (x_j in {0, 1})               │ clinical symptom present/absent lists. │
  ├─────────────────┼───────────────────────────────┼────────────────────────────────────────┤
  │ ComplementNB    │ Non-negative counts / TF-IDF  │ Severely imbalanced text (fraud detection,│
  │                 │ (x_j >= 0, heavily imbalanced)│ rare disease literature, content mod). │
  └─────────────────┴───────────────────────────────┴────────────────────────────────────────┘
```
