# 04. Gaussian Naive Bayes: Continuous Features & The Normal Distribution Assumption

Welcome to Guide 04 of the **Naive Bayes Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **the continuous probability density formulation, the Gaussian parameter mechanics, the variance-smoothing guard, and non-normal transformation pipelines**.

---

## 1. How Naive Bayes Handles Continuous Real Numbers

When classifying continuous clinical, physical, or financial measurements (e.g. `Patient Age`, `Blood Glucose`, `Tumor Radius`), we cannot count discrete occurrences.

Instead, **Gaussian Naive Bayes (`GaussianNB`)** models the likelihood of feature $x_j$ given class $c$ using the continuous **Gaussian (Normal) Probability Density Function (PDF)**:

$$P(x_j \mid y = c) = \frac{1}{\sqrt{2\pi \sigma_{cj}^2}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2} \right)$$

In log-space (used during inference for numerical stability):

$$\log P(x_j \mid y = c) = -\frac{1}{2} \ln(2\pi) - \ln(\sigma_{cj}) - \frac{(x_j - \mu_{cj})^2}{2\sigma_{cj}^2}$$

```
                           GAUSSIAN DENSITY CURVES PER CLASS
                 Probability
                   Density
                      ▲                 Class 0: Benign (μ=12, σ=2)
                      │                 Class 1: Malignant (μ=20, σ=3)
                      │             Class 0             Class 1
                      │               /\                  /\
                      │              /  \                /  \
                      │             /    \              /    \
                      │            /      \            /      \
                      │          ─/────────\──────────/────────\─
                      └──────────┼──────────┼─────────┼──────────┼────► Tumor Radius
                                10         12        17         20
```

---

## 2. Ultra-Lightweight Training: Only Two Numbers per Feature

While neural networks and tree ensembles learn thousands or millions of parameters, Gaussian Naive Bayes requires storing **only two scalar values per feature per class**:

1. **Class-Conditional Mean $\mu_{cj}$:**
   $$\mu_{cj} = \frac{1}{N_c} \sum_{i: y_i = c} x_{ij}$$
2. **Class-Conditional Variance $\sigma_{cj}^2$:**
   $$\sigma_{cj}^2 = \frac{1}{N_c} \sum_{i: y_i = c} (x_{ij} - \mu_{cj})^2$$

```
  MODEL STORAGE COMPARISON (30 Features, 2 Classes):
  ├── Random Forest (100 Trees, Depth 10) : ~15 MB
  ├── KNN Classifier (Training Data Store) : ~2.5 MB
  └── Gaussian Naive Bayes (2 * 30 * 2)   : ~120 FLOATS (< 1 KB!)
```

This makes `GaussianNB` virtually instantaneous to train ($O(N \cdot D)$) and execute in real-time embedded hardware, IoT devices, and microcontrollers.

---

## 3. The Zero-Variance Division Disaster & `var_smoothing`

### The Problem:
What happens if all training samples in class $c$ share the exact same value for feature $j$ (e.g., all benign patients have `num_pregnancies = 0`)?

$$\sigma_{cj}^2 = 0 \implies \frac{1}{\sqrt{2\pi \cdot 0}} \exp\left( -\frac{(x_j - \mu_{cj})^2}{0} \right) \implies \frac{1}{0} \implies \text{ZeroDivisionError / NaN!}$$

### The Solution: `var_smoothing`
Scikit-Learn introduces a mathematical safety floor called `var_smoothing`:

$$\hat{\sigma}_{cj}^2 = \sigma_{cj}^2 + \epsilon$$
$$\epsilon = \text{var\_smoothing} \times \max_{k, j}(\sigma_{kj}^2)$$

By default, `var_smoothing=1e-9`. It pads the empirical variance with a tiny fraction of the largest variance across all features, preventing infinite density spikes and numerical collapse.

```python
# Tuning var_smoothing across orders of magnitude
param_grid = {
    'gnb__var_smoothing': np.logspace(-11, -3, 9)
}
```

---

## 4. The Bell-Curve Trap: When Features Are NOT Gaussian

The primary Achilles' heel of `GaussianNB` is its dogmatic assumption that features follow a unimodal symmetric bell curve.

```
  GAUSSIAN ASSUMPTION SATISFIED               SKEWED / MULTIMODAL FAILURE
         (Works Flawlessly)                         (High Error Rate)
                /\                                 |\
               /  \                                | \
              /    \                               |  \
             /      \                              |   \_______
            /        \                             |           \_____
        Normal / Gaussian                           Log-Normal / Skewed
```

If real-world features are heavily skewed (e.g. `Income`, `Web Page Views`), exponential, or multi-modal:
1. The empirical mean $\mu$ is dragged into the tail by outliers.
2. The empirical variance $\sigma^2$ is artificially inflated, flattening the bell curve.
3. The model assigns completely erroneous likelihood probabilities.

### Enterprise Solution: Yeo-Johnson Power Transformation
Before passing continuous features to `GaussianNB`, transform them toward normality:

```python
from sklearn.preprocessing import PowerTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

gaussian_pipe = Pipeline([
    ('norm_transform', PowerTransformer(method='yeo-johnson')),
    ('gnb', GaussianNB())
])
```
* `PowerTransformer` automatically finds the optimal stabilizing parameter $\lambda$ to convert skewed, right-tailed, or left-tailed distributions into near-perfect bell curves!
