# ⚡ Algorithm 07: SGD Classifier (Large-Scale & Out-of-Core Classification)

> **The Senior Developer's Secret Weapon:**
> `SGDClassifier` Scikit-Learn ka sabse versatile linear classifier hai!
> Yeh ek "Chameleon" hai:
> * Agar aapne `loss='log_loss'` diya $\implies$ Yeh ban jata hai **Large-Scale Logistic Regression**!
> * Agar aapne `loss='hinge'` diya $\implies$ Yeh ban jata hai **Large-Scale Linear Support Vector Machine (SVM)**!
> * Agar aapne `loss='modified_huber'` diya $\implies$ Yeh ban jata hai **Outlier-Tolerant Probabilistic Classifier**!

---

## 📐 Mathematical Formulation

$$\text{Optimization Goal:} \quad \min_{\mathbf{w}} \frac{1}{m} \sum_{i=1}^m L\left(y^{(i)}, \mathbf{w}^T \mathbf{x}^{(i)} + b\right) + \alpha R(\mathbf{w})$$

* **Loss Function Equivalences (`loss=...`):**
  * `'log_loss'` $\to$ Logistic Regression (outputs calibrated `predict_proba`).
  * `'hinge'` $\to$ Maximum Margin Linear SVM (standard default).
  * `'modified_huber'` $\to$ Smooth zero-loss margin with probability estimates.

---

## 💻 Production Implementation (`SGDClassifier`)

```python
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# 1. Pipeline: Scaling (MANDATORY) + SGDClassifier
sgd_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SGDClassifier(
        loss='log_loss',         # Gives Logistic Regression behavior with predict_proba
        penalty='l2',
        alpha=0.0001,
        max_iter=1000,
        random_state=42
    ))
])

# 2. Fit on Training Data
sgd_clf.fit(X_train, y_train)

# 3. Evaluate Predictions & Probabilities
y_pred = sgd_clf.predict(X_test)
y_prob = sgd_clf.predict_proba(X_test)[:, 1]

print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"🎯 ROC-AUC Score : {roc_auc_score(y_test, y_prob):.4f}")
```

---

## 🧠 Universal Interview Flashcard: SGD Classifier

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"`SGDClassifier` aur `LogisticRegression` mein kya farq hai?"** | Alag algorithms hain. | `LogisticRegression(solver='lbfgs')` batch second-order optimization use karta hai (small to medium data ke liye perfect); `SGDClassifier(loss='log_loss')` first-order online stochastic gradient descent use karta hai jo **crores of rows** par fast scale hota hai. |
| **"`SGDClassifier` mein SVM kaise banate hain?"** | Naya import lagana padega. | Sirf `loss='hinge'` parameter pass karne se `SGDClassifier` Linear SVM ban jata hai! |

