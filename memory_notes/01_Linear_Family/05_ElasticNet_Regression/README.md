# ⚖️ Algorithm 05: Elastic Net Regression (Best of Both Worlds)

> **The Senior Developer's Golden Rule for Elastic Net:**
> Agar aapko **Lasso ki feature selection** bhi chahiye aur **Ridge ki multicollinearity stability** bhi chahiye, toh **Elastic Net** use karein!
> Yeh dono penalties ka convex combination banata hai. Agar 5 correlated features hain, toh Lasso unme se kisi ek ko randomly choose karegi aur baaki ko drop kar degi; jabki Elastic Net **poore correlated group ko ek saath retain** karta hai!

---

## 📐 Mathematical Formulation

$$\text{Loss}_{\text{ElasticNet}} = \frac{1}{2m} \sum_{i=1}^m \left(y^{(i)} - \hat{y}^{(i)}\right)^2 + \alpha \left( \rho \sum_{j=1}^n |w_j| + \frac{1 - \rho}{2} \sum_{j=1}^n w_j^2 \right)$$

* **`l1_ratio` ($\rho$):**
  * $\rho = 1.0 \implies$ Pure Lasso ($L_1$).
  * $\rho = 0.0 \implies$ Pure Ridge ($L_2$).
  * $0.0 < \rho < 1.0 \implies$ Elastic Net (Recommended: $0.5$ ya grid tune karein).

---

## 💻 Production Implementation (`ElasticNetCV`)

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Pipeline: Scaling + ElasticNet with 2D Grid Tuning (alpha & l1_ratio)
elastic_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('elastic', ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
        alphas=np.logspace(-3, 3, 50),
        cv=5,
        random_state=42
    ))
])

# 2. Fit on Training Data
elastic_pipeline.fit(X_train, y_train)

# 3. Best Hyperparameters
model = elastic_pipeline.named_steps['elastic']
print(f"🏆 Best Alpha (Overall Strength) : {model.alpha_:.4f}")
print(f"🏆 Best L1 Ratio (L1 vs L2 Mix)  : {model.l1_ratio_:.2f}")
print(f"📊 Test Set R² Score             : {elastic_pipeline.score(X_test, y_test):.4f}")
```

---

## 🧠 Universal Interview Flashcard: Elastic Net

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Elastic Net kab use karni chahiye?"** | Jab mann kare. | 1) Jab features number of samples se zyaada hon ($p > n$); 2) Jab features aapas mein heavily correlated hon (Lasso arbitrary selection ko prevent karne ke liye). |
| **"`l1_ratio` parameter kya control karta hai?"** | Learning rate. | Yeh $L_1$ aur $L_2$ ka ratio control karta hai ($\rho=1 \implies$ Lasso, $\rho=0 \implies$ Ridge, beech ki values mix balance banati hain). |

