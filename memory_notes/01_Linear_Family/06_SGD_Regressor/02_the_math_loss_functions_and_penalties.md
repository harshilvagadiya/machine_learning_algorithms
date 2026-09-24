# 📐 SGD Regressor Memory Notes: 02 - Math, Loss Functions & Regularization Penalties

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> SGDRegressor ek **Swiss Army Knife** hai!  
> Yeh akela class OLS, Ridge, Lasso, ElasticNet, aur yahan tak ki Outlier-Resistant Huber Regression — sab kuch ban sakta hai!  
> Sirf do arguments change karne hote hain:  
> - `loss`: Error kaise calculate karni hai (`squared_error`, `huber`, `epsilon_insensitive`).  
> - `penalty`: Weights ko kaise rokna hai (`'none'`, `'l2'`, `'l1'`, `'elasticnet'`).

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📐 Section 1: The General Weight Update Rule

Har individual sample $(\mathbf{x}_i, y_i)$ par, prediction hoti hai: $\hat{y}_i = \mathbf{w}^T \mathbf{x}_i + b$.

Weights update equation:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \left( 
abla_{\mathbf{w}} L(y_i, \hat{y}_i) + lpha 
abla_{\mathbf{w}} R(\mathbf{w}) ight)$$

Jahan:
- $\eta$ = Learning rate (step size).
- $
abla L$ = Loss function ka gradient (error direction).
- $lpha$ = Regularization penalty strength.
- $
abla R$ = Regularization penalty ka gradient.

---

## 🎯 Section 2: Supported Loss Functions (`loss=...`)

Scikit-Learn ka `SGDRegressor(loss='...')` 4 loss functions support karta hai:

### 1. `loss='squared_error'` (Standard OLS / MSE)
- Formula: $L(y_i, \hat{y}_i) = rac{1}{2} (y_i - \hat{y}_i)^2$
- Gradient: $
abla L = -(y_i - \hat{y}_i) \mathbf{x}_i$
- **Pros:** Mathematically optimal for Gaussian noise.
- **Cons:** Outliers ke square hone se gradients explode ho sakte hain.

### 2. `loss='huber'` (Outlier-Robust Regression)
- Error $|e| = |y_i - \hat{y}_i|$ ko check karta hai:
  $$L(e) = egin{cases} rac{1}{2} e^2 & 	ext{if } |e| \le \epsilon \ \epsilon |e| - rac{1}{2} \epsilon^2 & 	ext{if } |e| > \epsilon \end{cases}$$
- **Senior Developer Secret:** Agar data mein outliers hain aur aap unhe drop nahi kar sakte, toh `loss='huber'` use karein! Chhote errors quadratic rehte hain aur bade errors linear ban jaate hain, jisse model outliers ke aage jhukta nahi hai!

### 3. `loss='epsilon_insensitive'` (Support Vector Regression - SVR)
- Agar error $\le \epsilon$ hai, toh loss = 0! Sirf $\epsilon$-tube ke bahar ke errors par penalty lagti hai.

---

## 🛡️ Section 3: Supported Penalties (`penalty=...`)

Aap SGDRegressor ko kisi bhi standard linear model mein convert kar sakte ho:

| Parameter Settings | Equivalent Classical Algorithm | Purpose |
| :--- | :--- | :--- |
| `loss='squared_error'`, `penalty=None` | **OLS Linear Regression** | Standard unpenalized gradient descent |
| `loss='squared_error'`, `penalty='l2'` | **Ridge Regression ($L_2$)** | Multicollinearity control (Default!) |
| `loss='squared_error'`, `penalty='l1'` | **Lasso Regression ($L_1$)** | Automatic feature sparsity |
| `loss='squared_error'`, `penalty='elasticnet'` | **ElasticNet ($L_1 + L_2$)** | Grouping + Sparsity (`l1_ratio` parameter ke sath) |
| `loss='huber'`, `penalty='l2'` | **Robust Ridge Regression** | Heavy outliers + Multicollinearity |

---

## ⚡ 10-Second Interview Flashcard: Loss & Penalty

| Question | Junior Developer Answer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"SGDRegressor mein Ridge aur Lasso kaise switch karte hain?"** | Alag-alag classes import karke. | `penalty='l2'` se Ridge ban jata hai aur `penalty='l1'` se Lasso ban jata hai. Dono ka mix `penalty='elasticnet'` se banta hai. |
| **"Outliers se deal karne ke liye SGDRegressor mein kaun si loss best hai?"** | squared_error hi chalegi. | `loss='huber'` best hai, kyunki wo threshold $\epsilon$ ke baad loss ko linear kar deta hai, jisse outliers ka gradient finite rehta hai aur weights explode nahi hote. |
