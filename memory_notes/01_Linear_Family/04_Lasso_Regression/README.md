# ✂️ Algorithm 04: Lasso Regression (L1 Regularization & Feature Selection)

> **The Senior Developer's Golden Rule for Lasso:**
> Agar aapke paas **hazaron features** hain (High Dimensionality, e.g., Genomics, Click-Stream, Text features) aur aapko pata hai ki unme se zyadatar useless hain, toh **Lasso ($L_1$)** use karein!
> Lasso loss function mein **Absolute Weights Penalty ($\alpha \sum |w_j|$)** lagata hai, jiska geometrical diamond corner non-informative weights ko **exact ZERO** bana deta hai!

---

## 📐 Mathematical Formulation

$$\text{Loss}_{\text{Lasso}} = \frac{1}{2m} \sum_{i=1}^m \left(y^{(i)} - \hat{y}^{(i)}\right)^2 + \alpha \sum_{j=1}^n |w_j|$$

* **Geometric Intuition (Kyu Exact Zero Banta Hai?):**
  * $L_2$ (Ridge) constraint circle/sphere hoti hai $\implies$ Contours tangents par milte hain jahan $w \neq 0$.
  * $L_1$ (Lasso) constraint diamond/polytope hoti hai jisme **teekhe corners (sharp vertices) axes par** hote hain $\implies$ Contours aksar axis ke corner par hit karte hain, jisse weight exact $0.0$ ho jata hai!

---

## 💻 Production Implementation (`LassoCV`)

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Pipeline: Scaling (MANDATORY) + Lasso with CV
lasso_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, random_state=42))
])

# 2. Fit on Training Data
lasso_pipeline.fit(X_train, y_train)

# 3. Inspect Selected vs Dropped Features
lasso_model = lasso_pipeline.named_steps['lasso']
zero_weights = np.sum(lasso_model.coef_ == 0)
total_features = len(lasso_model.coef_)

print(f"🏆 Best L1 Penalty Alpha  : {lasso_model.alpha_:.4f}")
print(f"✂️ Features Eliminated   : {zero_weights} / {total_features} ({(zero_weights/total_features)*100:.1f}%)")
print(f"📊 Test Set R² Score      : {lasso_pipeline.score(X_test, y_test):.4f}")
```

---

## 🧠 Universal Interview Flashcard: Lasso Regression

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Lasso aur Ridge mein main difference kya hai?"** | Dono penalty lagate hain. | Ridge ($L_2$) weights ko chhota karta hai par zero nahi karta; Lasso ($L_1$) irrelevant weights ko **exact $0.0$** karke **automatic feature selection** karta hai (Sparse Model). |
| **"Lasso kab fail ho sakti hai?"** | Kabhi fail nahi hoti. | Agar group of correlated features ho (e.g. 5 collinear features), toh Lasso kisi ek ko randomly choose karegi aur baaki 4 ko drop kar degi. Aise case mein **Elastic Net** use karte hain. |

