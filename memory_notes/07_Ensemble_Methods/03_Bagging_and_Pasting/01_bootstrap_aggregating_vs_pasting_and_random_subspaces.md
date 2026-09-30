# 🎒 Bagging, Pasting, Random Subspaces & Random Patches

Bootstrap Aggregation (Bagging; Leo Breiman, 1996) is one of the most foundational variance-reduction ensemble methods in statistical machine learning.

---

## 1. The Variance Reduction Mathematics of Bagging

Consider an ensemble of $B$ base estimators $\{h_b(x)\}_{b=1}^B$, each having variance $\sigma^2$ and positive pairwise correlation $\rho = \text{Corr}(h_i(x), h_j(x))$.

The variance of their simple average $\hat{y}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} h_b(x)$ is given by:

$$\text{Var}(\hat{y}_{\text{bag}}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

### Key Insights:
1. As the number of estimators $B \to \infty$, the term $\frac{1 - \rho}{B} \sigma^2 \to 0$.
2. The remaining lower bound on variance is **$\rho \sigma^2$** (governed entirely by tree correlation!).
3. Bagging never increases the bias of base models; it strictly suppresses variance.
4. Hence, base learners in bagging should be **complex, low-bias, high-variance models** (e.g., deep unpruned decision trees).

---

## 2. Bagging vs. Pasting vs. Random Subspaces vs. Random Patches

| Ensemble Strategy | Row Sampling (Observations) | Column Sampling (Features) | Replacement (`bootstrap`) | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Bagging** | Subset ($\approx 63.2\%$) | Full ($100\%$) | `bootstrap=True` | Standard variance reduction |
| **Pasting** | Subset ($< 100\%$) | Full ($100\%$) | `bootstrap=False` | Massive datasets exceeding RAM |
| **Random Subspaces** | Full ($100\%$) | Subset ($< 100\%$) | `bootstrap_features=True/False` | High-dimensional gene/text data |
| **Random Patches** | Subset ($< 100\%$) | Subset ($< 100\%$) | Both | Large, high-dimensional datasets |

### The Out-Of-Bag (OOB) Statistical Wonder
When drawing $N$ samples with replacement from a dataset of size $N$, the probability of an observation NOT being chosen in a given draw is $1 - \frac{1}{N}$.
For $N$ draws:

$$\lim_{N \to \infty} \Big(1 - \frac{1}{N}\Big)^N = \frac{1}{e} \approx 0.368 = 36.8\%$$

- Roughly **36.8%** of samples are never selected for a given tree. These are the **Out-Of-Bag (OOB)** samples.
- By evaluating each tree on its own OOB samples, we obtain an unbiased generalization score **without requiring an external cross-validation split** (`oob_score=True`)!

---

## 3. Simple vs. Enterprise Production Code

### 3.1 Simple Example (Standard Bagging with OOB)
```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bag_clf.fit(X_train, y_train)
print(f"OOB Accuracy: {bag_clf.oob_score_:.4f}")
```

### 3.2 Enterprise Production Random Patches Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Random Patches: max_samples=0.7 (subsample rows), max_features=0.8 (subsample columns)
bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=10),
    n_estimators=100,
    max_samples=0.7,
    max_features=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

enterprise_pipeline = Pipeline([
    ('prep', preprocessor),
    ('bagging', bag_reg)
])
enterprise_pipeline.fit(X_train, y_train)
```
