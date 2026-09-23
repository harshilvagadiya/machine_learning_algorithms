# 🎯 Classification Memory Notes: 09 - Logistic Regression Algorithm, Loss Function & Hyperparameter Tuning

> **The Optimization Secret:**
> Linear Regression ek sidhi lathi (straight line) thi jo $-\infty$ se $+\infty$ tak jaati thi.
> Lekin Probability hamesha **0 se 1 (0% se 100%)** ke beech bandhi hoti hai! 
> Is sidhi lathi ko 0 aur 1 ke beech modhne ke liye hum laate hain **Sigmoid Function**.
> Aur Linear Regression ka Mean Squared Error (MSE) yahan fail ho jata hai kyunki wo pahadon jaisi wavy curve (Non-Convex) banata hai jisme Gradient Descent phans jata hai. Isliye hum use karte hain **Binary Cross-Entropy (Log Loss)** jo ekdum katora-jaisi (Convex) hoti hai!

---

## 🧭 The Optimization Roadmap

```
                          LOGISTIC REGRESSION ENGINE
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           ▼                                                     ▼
ALGORITHM & LOSS FUNCTION                                 HYPERPARAMETER TUNING
- Why Linear fails (Outlier shift)                       - Inverse Lambda: C = 1 / λ
- Sigmoid: σ(z) = 1 / (1 + e^-z)                         - Solvers: lbfgs, saga, liblinear
- Binary Cross-Entropy (Strictly Convex)                 - Regularization: l1 (Lasso), l2 (Ridge)
- Infinite penalty for arrogant mistakes                 - Class Weight: None vs balanced
```

---

## 🚫 Section 1: Why Linear Regression Fails for Classification

### Problem 1: Predictions Outside [0, 1]
Agar binary target ($y \in \{0, 1\}$) par sidha OLS Linear Regression fit karein: $\hat{y} = w \cdot x + b$:
- Agar feature $x$ bohot bada ho gaya, linear equation predict karegi $\hat{y} = 2.3$ (230% chance? Yeh impossible hai!).
- Agar feature $x$ bohot chhota hua, predict karegi $\hat{y} = -0.4$ (-40% chance?).

### Problem 2: The Outlier Shift Catastrophe
Linear regression squared error ($e^2$) minimize karta hai. Agar ek bohot bada outlier aa gaya:
- Linear Regression ki poori line outlier ki taraf jhuk jaati hai.
- Is jhukaav ki wajah se **0.5 threshold shift ho jata hai**, aur normal applicants bhi galat classify hone lagte hain!

---

## 🌊 Section 2: The Sigmoid Curve & Odds

Linear equation $z = w^T x + b$ ko **Sigmoid (Logistic) Function** ke andar pass karte hain:

$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(w_1 x_1 + \dots + w_n x_n + b)}}$$

```
    P(y = 1)
       1.0 |                             ●---●---● (Class 1)
           |                           /
       0.5 |------------------------ / ----------- (Decision Boundary at z = 0)
           |                       /
       0.0 | ●---●---● (Class 0) /
           +-------------------------+------------> z = w*x + b
                                   z = 0
```

* Jab $z \to +\infty \implies e^{-z} \to 0 \implies \sigma(z) \to 1.0$ (100% Probability)
* Jab $z = 0 \implies e^0 = 1 \implies \sigma(0) = \frac{1}{1 + 1} = 0.50$ (50% Boundary)
* Jab $z \to -\infty \implies e^{-z} \to +\infty \implies \sigma(z) \to 0.0$ (0% Probability)

### Probability $\to$ Odds $\to$ Logit Link Function
$$\text{Odds} = \frac{p}{1 - p} \implies \ln(\text{Odds}) = \ln\left(\frac{p}{1 - p}\right) = w^T x + b$$
Yeh formula batata hai ki Logistic Regression असल mein features ke linear combination se **Log-Odds** predict karta hai!

---

## 📉 Section 3: Why MSE Fails & Binary Cross-Entropy (Log Loss)

Linear Regression mein $y = wx + b$ linear tha, isliye MSE surface ek perfect katora (Paraboloid) tha.
Lekin Logistic Regression mein Sigmoid non-linear hai! Agar MSE use kiya:

```
   MSE FOR LOGISTIC REGRESSION (NON-CONVEX)          LOG LOSS / CROSS-ENTROPY (CONVEX)
             /\          /\                                       \               /
            /  \  Local /  \                                       \             /
           /    \/Minima\/  \                                       \   Global  /
          /                  \                                       \  Minimum/
         +--------------------+                                       +-------+
       Gradient Descent gets TRAPPED!                         Always reaches Global Minimum!
```

### The Solution: Binary Cross-Entropy Formula
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \ln(p^{(i)}) + (1 - y^{(i)}) \ln(1 - p^{(i)}) \right]$$

* **Jab Actual Class $y = 1$ ho:** $\text{Loss} = -\ln(p)$. Agar model ne $p = 0.01$ bola, toh $-\ln(0.01) \approx 4.60$ (Bhari penalty!). Agar $p \to 0$, toh Loss $\to +\infty$ (**Infinite Punishment**).
* **Jab Actual Class $y = 0$ ho:** $\text{Loss} = -\ln(1 - p)$. Agar model ne $p = 0.99$ bola, toh Loss $\to +\infty$!

---

## 🎛️ Section 4: Hyperparameter Tuning Suite

### 1. The `C` Parameter (Inverse Regularization Strength)
$$C = \frac{1}{\lambda}$$

| Value of $C$ | Regularization Strength | Model Complexity | Risk |
| :--- | :--- | :--- | :--- |
| **$C = 0.01$** | Bohot Strong (Chhote weights) | Simple, High Bias | Underfitting |
| **$C = 1.0$ (Default)** | Balanced | Balanced | Sweet spot for most data |
| **$C = 100$** | Bohot Weak (Bade weights) | Complex, High Variance | Overfitting |

### 2. Choosing the Right Solver (`solver=...`)
| Solver | Penalty Supported | Dataset Size | Best Used For |
| :--- | :--- | :--- | :--- |
| **`lbfgs`** *(Default)* | L2, None | Small to Medium | Tabular data, fast convergence (Standard choice) |
| **`liblinear`** | L1, L2 | Small | Chhota data jisme L1 feature selection chahiye |
| **`saga`** | L1, L2, ElasticNet, None | Large Datasets | Big Data, sparse matrices, ElasticNet |

### 3. Penalty (`penalty=...`)
* **`'l2'` (Ridge - Default):** Saare weights ko chhota karta hai, zero nahi karta.
* **`'l1'` (Lasso):** Faltu features ke weights ko seedha $0$ kar deta hai (Feature Selection).

### 4. Class Weight (`class_weight=...`)
* **`None` (Default):** Balanced datasets ke liye.
* **`'balanced'`:** Imbalanced datasets (e.g., 95% vs 5%) ke liye.

---

## 🎯 Section 5: Multiclass Classification (OvR vs Multinomial)

Jab target mein 3 ya usse zyada classes hon (jaise Low, Medium, High):
1. **One-vs-Rest (OvR):** 3 classes ke liye model 3 alag binary models train karta hai (Class A vs B+C, Class B vs A+C, Class C vs A+B).
2. **Multinomial (Softmax Regression):** Teeno classes ke liye ek sath Softmax formula lagata hai taaki probabilities ka sum exactly $1.0$ (100%) aaye:
$$P(y = k | X) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

---

## 💻 Section 6: Production GridSearchCV Pattern

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear'],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000, random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',   # Imbalanced data ke liye accuracy nahi, ROC-AUC ya F1!
    cv=5,
    n_jobs=-1
)

grid.fit(X_train_final, y_train)

print(f"🏆 Best Hyperparameters : {grid.best_params_}")
print(f"📊 Best CV ROC-AUC Score: {grid.best_score_:.4f}")
```

---

## 🧠 Quick Revision Checklist: Loss & Tuning
- [ ] Pata hai kyu MSE Logistic Regression ke liye non-convex hota hai?
- [ ] Log Loss ka formula aur infinite penalty ka logic clear hai?
- [ ] Pata hai ki $C = \frac{1}{\lambda}$ hota hai (Low $C$ = High Regularization)?
- [ ] L1 penalty ke liye `solver='liblinear'` ya `solver='saga'` select kiya?
