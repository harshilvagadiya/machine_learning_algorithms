# ⚡ Algorithm 06: SGD Regressor (Large-Scale & Out-of-Core Regression)

> **The Senior Developer's Golden Rule for SGD Regressor:**
> Standard Linear Regression (`LinearRegression`) aur Ridge poore dataset ko RAM mein load karke matrix operations ($(\mathbf{X}^T \mathbf{X})^{-1}$) karti hain.
> Agar dataset **10 Lakh se 1 Crore rows** ka hai, toh RAM turant Out-Of-Memory (OOM) crash ho jayegi!
> **SGDRegressor** Stochastic Gradient Descent use karta hai — har sample ke sath weights ko iteratively update karta hai. Yeh **Out-of-Core learning (`partial_fit`)** aur real-time streaming data ke liye industry standard hai!

---

## 📐 Mathematical Formulation

Har single sample $(x^{(i)}, y^{(i)})$ ke liye weights update hote hain:

$$w_j \leftarrow w_j - \eta \left( \frac{\partial \text{Loss}}{\partial w_j} + \alpha \frac{\partial \text{Penalty}}{\partial w_j} \right)$$

* **Loss Functions Supported (`loss=...`):**
  * `'squared_error'` (Default OLS regression).
  * `'huber'` (Outlier robust loss — switches from quadratic to linear for errors $> \epsilon$).
  * `'epsilon_insensitive'` (Support Vector Regression loss).
* **Penalties Supported (`penalty=...`):** `'l2'`, `'l1'`, `'elasticnet'`.

---

## 💻 Production Implementation (`SGDRegressor`)

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Pipeline: Scaling is 100% NON-NEGOTIABLE for SGD (Gradient Descent fails if unscaled!)
sgd_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SGDRegressor(
        loss='squared_error',
        penalty='l2',
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        learning_rate='invscaling',
        eta0=0.01,
        random_state=42
    ))
])

# 2. Fit on Training Data
sgd_reg.fit(X_train, y_train)

print(f"📊 Test Set R² Score: {sgd_reg.score(X_test, y_test):.4f}")
```

---

## 🧠 Universal Interview Flashcard: SGD Regressor

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"SGDRegressor kab use karte hain?"** | Normal linear regression ki jagah. | Jab dataset **bohot bada ho (Big Data > 1M rows)** jo RAM mein fit na ho sake, ya jab data **streaming / batch-by-batch (`partial_fit`)** aa raha ho. |
| **"SGD ke liye feature scaling kyu compulsory hai?"** | Sirf acche results ke liye. | SGD gradient descent par chalta hai. Agar feature scale nahi hua toh gradients oscillate karenge aur model explode/diverge ho jayega! |

