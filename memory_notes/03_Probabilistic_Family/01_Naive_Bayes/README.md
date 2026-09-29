# 🔵 Probabilistic Family: Naive Bayes Classifiers

Welcome to the **Naive Bayes** architectural knowledge hub.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard end-to-end ML lifecycle steps (Data ingestion, missing value imputation, EDA, train/test splitting, ColumnTransformer preprocessing, and evaluation metrics) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> **The guides below document ONLY the novel mathematical, probabilistic, and operational concepts unique to Naive Bayes.**

---

## 📚 Complete Naive Bayes Guide Suite

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                    PROBABILISTIC CLASSIFICATION MAP                    │
  └────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
   Foundational Theory                                    Engineering & Estimators
   ├── 01. Bayes' Theorem & Posterior Formulation         ├── 04. Gaussian NB (Continuous Features)
   ├── 02. The "Naive" Independence Assumption            ├── 05. Multinomial NB (NLP & Count Vectors)
   └── 03. The Zero-Frequency Disaster & Laplace          ├── 06. Bernoulli & Complement NB (Imbalance)
                                                          └── 07. Production Pipelines & Streaming SLAs
```

| Guide | Core Focus | Key Questions Answered |
| :--- | :--- | :--- |
| **[01. Bayes' Theorem & Posteriors](01_bayes_theorem_prior_likelihood_posterior.md)** | Core Probability Theory | Prior vs. Likelihood vs. Evidence. Why log-space sums prevent floating-point underflow. |
| **[02. The "Naive" Assumption](02_the_naive_conditional_independence_assumption.md)** | Independence Mechanics | Why does Naive Bayes achieve high classification accuracy even when features are strongly correlated? |
| **[03. Zero-Frequency & Laplace](03_the_zero_frequency_problem_and_laplace_smoothing.md)** | Failure Modes & Smoothing | How a single unseen token wipes out likelihood to zero, and how Laplace $\alpha$ fixes it. |
| **[04. Gaussian NB & Bell Curves](04_gaussian_nb_continuous_features_and_normal_distribution.md)** | Continuous Features | How continuous Gaussian PDFs work, parameter storage (only 2 numbers per feature), and `var_smoothing`. |
| **[05. Multinomial NB & Text NLP](05_multinomial_nb_count_vectors_and_text_classification.md)** | Count & TF-IDF Vectors | Bag-of-words PMF, why fractional TF-IDF works, and the fatal negative-value scaler crash. |
| **[06. Bernoulli & Complement NB](06_bernoulli_and_complement_nb_for_imbalanced_data.md)** | Binary & Imbalanced Data | The penalty of absence in Bernoulli trials, and why ComplementNB conquers skewed class ratios. |
| **[07. Production Pipelines & SLAs](07_production_pipeline_vectorization_and_live_inference.md)** | Production Deployment | Out-of-core learning with streaming `partial_fit`, joblib serialization, and sub-millisecond SLAs. |

---

## ⚡ The Naive Bayes Cheat Sheet

* **Classification Rule:** $\hat{y} = \arg\max_{c} \left[ \log P(y=c) + \sum_{j=1}^D \log P(x_j \mid y=c) \right]$.
* **MultinomialNB Constraint:** Strictly requires non-negative features ($x_j \ge 0$). **Never use `StandardScaler()`**.
* **Zero Frequency Guard:** Always enable additive smoothing ($\alpha > 0$) on text or discrete count data.
* **Gaussian Variance Guard:** Keep `var_smoothing >= 1e-9` to avoid zero-division on uniform continuous columns.
* **Inference Speed:** Pure vector dot product — among the fastest classifiers in computing history ($< 0.5\text{ms}$).

---

## 🧪 Enterprise Production Notebooks Suite (4 Executed Implementations)

The companion practical notebooks in [`supervised_learning/NaiveBayes/`](file:///home/python/03_Custom_addons/ML/supervised_learning/NaiveBayes/) implement real-world end-to-end applications across diverse domains:

1. **[SMS Spam Detection (MultinomialNB)](file:///home/python/03_Custom_addons/ML/supervised_learning/NaiveBayes/Spam_SMS_Detection_MultinomialNB.ipynb)**
   * *Domain:* Cyber-Security & Natural Language Processing
   * *Estimator:* `MultinomialNB` with `TfidfVectorizer(sublinear_tf=True)`
   * *Model:* `spam_multinomial_nb_pipeline.joblib`
   * *Result:* ROC-AUC $0.9925$, F1-Score $93.60\%$, Latency $1.2\text{ms}$
2. **[Breast Cancer Diagnostic Screening (GaussianNB)](file:///home/python/03_Custom_addons/ML/supervised_learning/NaiveBayes/Breast_Cancer_Diagnostic_GaussianNB.ipynb)**
   * *Domain:* Clinical Oncology & Biometric Diagnostic
   * *Estimator:* `GaussianNB` with `PowerTransformer(yeo-johnson)`
   * *Model:* `breast_cancer_gaussian_nb_pipeline.joblib`
   * *Result:* Accuracy $94.74\%$, ROC-AUC $0.9884$, Size $4.1\text{ KB}$
3. **[Wine Cultivar Chemical Classification (Multiclass GaussianNB)](file:///home/python/03_Custom_addons/ML/supervised_learning/NaiveBayes/Wine_Cultivar_Multiclass_GaussianNB.ipynb)**
   * *Domain:* Chemical Agronomy & Beverage Quality Assurance
   * *Estimator:* `GaussianNB` (3 Cultivar classes, continuous chemical features)
   * *Model:* `wine_multiclass_gaussian_nb_pipeline.joblib`
   * *Result:* Accuracy $97.22\%$, Macro-F1 $97.22\%$, Size $2.6\text{ KB}$
4. **[Customer Product Sentiment Analysis (BernoulliNB)](file:///home/python/03_Custom_addons/ML/supervised_learning/NaiveBayes/Customer_Sentiment_BernoulliNB.ipynb)**
   * *Domain:* E-Commerce Feedback & Binary Lexicon Presence
   * *Estimator:* `BernoulliNB` with `CountVectorizer(binary=True)`
   * *Model:* `sentiment_bernoulli_nb_pipeline.joblib`
   * *Result:* Accuracy $100.00\%$, ROC-AUC $1.0000$, Sub-millisecond latency
