# 01. Bayes' Theorem: Prior, Likelihood, and Posterior Formulation

Welcome to the probabilistic foundation of the **Naive Bayes Classifier**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **the foundational probabilistic mechanics, Bayes' Rule derivations, and the generative modeling paradigm**.

---

## 1. The Core Probability Engine: Bayes' Theorem

At its mathematical heart, Naive Bayes answers a simple, fundamental question:
**"Given an observed vector of evidence $\mathbf{x} = [x_1, x_2, \dots, x_D]^T$, what is the probability that it belongs to target class $y = c$?"**

Bayes' Theorem provides the exact formula to invert conditional probabilities:

$$P(y = c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y = c) \cdot P(y = c)}{P(\mathbf{x})}$$

```
                ┌────────────────────────────────────────────────────────┐
                │                     BAYES' THEOREM                     │
                └────────────────────────────────────────────────────────┘
                                            │
           ┌────────────────────────────────┼────────────────────────────────┐
           ▼                                ▼                                ▼
      POSTERIOR                         LIKELIHOOD                         PRIOR
   P(y = c | x)                     P(x | y = c)                         P(y = c)
"What is probability              "If class is c,                   "Baseline belief
 of class c given                 how likely are                     before seeing
 observed features?"              these features?"                   any features"
           │                                │                                │
           └────────────────────────────────┼────────────────────────────────┘
                                            ▼
                                        EVIDENCE
                                          P(x)
                              "Total probability of x
                               across all possible classes"
```

---

## 2. Anatomy of the Four Components

### A. The Prior Probability $P(y = c)$
* **Meaning:** The baseline background rate of class $c$ in the absence of any feature information.
* **Calculation:**
  $$P(y = c) = \frac{N_c}{N_{\text{total}}}$$
  where $N_c$ is the count of samples in class $c$, and $N_{\text{total}}$ is the total training dataset size.
* **Real-world Example:** If $15\%$ of incoming company emails are spam, $P(y = \text{spam}) = 0.15$ and $P(y = \text{ham}) = 0.85$.

### B. The Likelihood $P(\mathbf{x} \mid y = c)$
* **Meaning:** The probability of generating feature values $\mathbf{x}$ assuming that the sample genuinely belongs to class $c$.
* **Challenge:** Directly estimating the joint conditional probability $P(x_1, x_2, \dots, x_D \mid y = c)$ in a high-dimensional space requires exponential sample sizes (the curse of dimensionality). This is where the *Naive Assumption* steps in (detailed in Guide 02).

### C. The Marginal Evidence $P(\mathbf{x})$
* **Meaning:** The total probability of observing the feature vector $\mathbf{x}$ across the entire sample space.
* **Law of Total Probability:**
  $$P(\mathbf{x}) = \sum_{k=1}^{K} P(\mathbf{x} \mid y = k) \cdot P(y = k)$$
* **Optimization Property:** Because $P(\mathbf{x})$ is identical for all candidate classes $k \in \{1, \dots, K\}$, it acts purely as a normalizing constant to ensure $\sum_{c=1}^K P(y = c \mid \mathbf{x}) = 1$. When selecting the most probable class, **$P(\mathbf{x})$ can be completely ignored**!

### D. The Posterior Probability $P(y = c \mid \mathbf{x})$
* **Meaning:** The revised, updated belief after synthesizing prior knowledge with observed empirical evidence.
* **Decision Rule (MAP - Maximum A Posteriori):**
  $$\hat{y} = \arg\max_{c \in \{1, \dots, K\}} P(y = c \mid \mathbf{x})$$

---

## 3. Generative vs. Discriminative Paradigm

Machine learning classifiers fall into two distinct philosophical camps:

```
  ┌────────────────────────────────────────┬────────────────────────────────────────┐
  │         DISCRIMINATIVE MODELS          │           GENERATIVE MODELS            │
  │     (Logistic Regression, SVM)         │             (Naive Bayes)              │
  ├────────────────────────────────────────┼────────────────────────────────────────┤
  │ Directly models the boundary:          │ Models how each class generates data:  │
  │          P(y | x)                      │           P(x, y) = P(x | y) P(y)      │
  │ Ignores how x was generated; focuses   │ Learns the full joint probability      │
  │ purely on separating class clouds.     │ distribution of features within class. │
  │ Higher asymptotic accuracy when        │ Faster convergence with small data     │
  │ sample size N -> infinity.             │ (O(log D) samples vs O(D) for LogReg). │
  └────────────────────────────────────────┴────────────────────────────────────────┘
```

> **Why Generative Matters in Production:**
> Because Naive Bayes learns $P(\mathbf{x} \mid y)$, it can naturally handle missing attributes during inference: simply marginalize out (drop) the missing feature from the product without requiring imputer pipelines for evaluation!

---

## 4. The Log-Space Transformation: Preventing Floating-Point Underflow

In production code, **NEVER multiply raw probabilities directly**. 

When computing the likelihood across hundreds or thousands of features (such as vocabulary terms in NLP):
$$P(\mathbf{x} \mid y = c) = \prod_{j=1}^D P(x_j \mid y = c)$$

Each individual probability $P(x_j \mid y = c)$ is a float between $0$ and $1$ (e.g. $0.001$). Multiplying $1,000$ small floats causes instantaneous IEEE-754 **floating-point underflow to zero** (`0.0`), destroying model predictions.

### The Mathematical Fix: Add Logarithms
Taking the natural logarithm transforms the product into a computationally stable summation:

$$\log P(y = c \mid \mathbf{x}) \propto \log P(y = c) + \sum_{j=1}^{D} \log P(x_j \mid y = c)$$

$$\hat{y} = \arg\max_{c \in \{1, \dots, K\}} \left[ \log P(y = c) + \sum_{j=1}^{D} \log P(x_j \mid y = c) \right]$$

* Summing negative log-probabilities preserves high precision.
* Monotonicity: Since $\log(z)$ is strictly increasing, $\arg\max \log f(z) = \arg\max f(z)$.

---

## 5. Summary Cheat Sheet

| Symbol / Concept | Term | Operational Purpose |
| :--- | :--- | :--- |
| $P(y = c)$ | **Prior** | Frequency of class $c$ in training set |
| $P(x_j \mid y = c)$ | **Likelihood** | Feature distribution given class (Gaussian, Multinomial, Bernoulli) |
| $P(\mathbf{x})$ | **Evidence** | Constant denominator, discarded during $\arg\max$ classification |
| $P(y = c \mid \mathbf{x})$ | **Posterior** | Final confidence probability used for predictions |
| **Log-Sum-Exp** | Numerical Stability | Prevents underflow to $0.0$ when multiplying small probabilities |
