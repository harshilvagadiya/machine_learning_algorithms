# 🗳️ Voting Ensembles: Hard vs. Soft Voting Principles & Theory

Voting Ensembles combine predictions from multiple distinct, diverse machine learning algorithms (e.g., Logistic Regression, Support Vector Classifiers, Random Forests, KNN) into a single aggregated prediction.

---

## 1. Theoretical Foundation & Condorcet's Jury Theorem

The power of voting ensembles originates in statistical jury theorems:
If you aggregate the decisions of $M$ independent voters, each having an individual accuracy $p > 0.5$, the probability $P_{\text{ensemble}}$ that the majority vote is correct approaches $1$ as $M \to \infty$:

$$P_{\text{ensemble}} = \sum_{k=\lfloor M/2 \rfloor + 1}^{M} \binom{M}{k} p^k (1 - p)^{M - k}$$

### Key Practical Caveat: Error Independence
- The theorem strictly assumes that voter errors are **uncorrelated** ($\rho = 0$).
- If models share similar architectures or make identical mistakes on the same sub-regions, the ensemble will not yield significant variance reduction.
- **Golden Rule:** Combine structurally diverse model families (e.g., Linear + Distance-Based + Tree-Based).

---

## 2. Hard Voting vs. Soft Voting

### 2.1 Hard Voting (Majority Rule)
In hard voting, each base estimator predicts a discrete class label $\hat{y}_m \in \{1, \dots, C\}$. The ensemble selects the class receiving the absolute plurality of votes:

$$\hat{y}_{\text{hard}} = \text{mode} \big\{ \hat{y}_1, \hat{y}_2, \dots, \hat{y}_M \big\} = \arg\max_{c} \sum_{m=1}^{M} \mathbb{I}(\hat{y}_m = c)$$

- **Pros:** Does not require well-calibrated class probability estimates; works with any classifier (e.g., SVM with linear kernel, Perceptron).
- **Cons:** Discards prediction confidence. A model that predicts class 1 with 51% certainty has the same weight as a model predicting class 1 with 99.9% certainty.

### 2.2 Soft Voting (Weighted Probability Average)
In soft voting, each base estimator outputs continuous class probability vectors $P_m(y = c \mid x)$. The ensemble computes the weighted average probability across all base estimators:

$$\hat{P}_{\text{soft}}(y = c \mid x) = \frac{\sum_{m=1}^{M} w_m P_m(y = c \mid x)}{\sum_{m=1}^{M} w_m}$$
$$\hat{y}_{\text{soft}} = \arg\max_{c} \hat{P}_{\text{soft}}(y = c \mid x)$$

- **Pros:** Significantly outperforms hard voting because confident, well-calibrated models have higher influence on borderline cases.
- **Requirement:** All base estimators must implement `predict_proba()` with calibrated probabilities. For SVMs, set `probability=True` (Platt scaling).

---

## 3. Simple Example vs. Enterprise Production Example

### 3.1 Simple Example (Hard vs Soft Voting)
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Define base estimators
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = GaussianNB()

# Soft Voting Ensemble
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='soft',
    weights=[1, 2, 1]  # RF given double weight
)
eclf.fit(X_train, y_train)
preds = eclf.predict(X_test)
```

### 3.2 Enterprise Production Example (Voting with Preprocessing Pipelines)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])

# Encapsulated Base Pipelines (preventing leakage & scale mismatch)
pipe_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(C=1.0, max_iter=1000))])
pipe_rf = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, max_depth=8))])
pipe_svc = Pipeline([('prep', preprocessor), ('clf', SVC(C=1.0, kernel='rbf', probability=True))])

voting_ensemble = VotingClassifier(
    estimators=[
        ('lr', pipe_lr),
        ('rf', pipe_rf),
        ('svc', pipe_svc)
    ],
    voting='soft',
    weights=[1.0, 1.5, 1.2],
    n_jobs=-1
)
```

---

## 4. Voting Regressor Formulation
For continuous targets, `VotingRegressor` averages the predictions of base regressors:

$$\hat{y}_{\text{vote}}(x) = \frac{\sum_{m=1}^{M} w_m \hat{y}_m(x)}{\sum_{m=1}^{M} w_m}$$

- Ideal for smoothing regression discontinuities when combining tree regressors with linear and spline models.
