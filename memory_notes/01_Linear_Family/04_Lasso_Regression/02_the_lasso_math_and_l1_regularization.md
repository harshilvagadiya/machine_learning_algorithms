# ✂️ Lasso Regression Memory Notes: 02 - The Math & $L_1$ Regularization

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Ridge Regression ka formula smooth tha kyunki usme square ($w^2$) tha (smooth bowl shape).  
> Lekin Lasso mein **Absolute Value ($|w|$)** hota hai.  
> Aur $|w|$ ka graph kaisa dikhta hai? **V-Shape (ek sharp chaku jaisi nok $w=0$ par)!**  
> School ka calculus bolta hai: *"Chaku ki nok par tangent (derivative) draw nahi ho sakta!"*  
> Toh computer Lasso ko solve kaise karta hai? **Coordinate Descent & Soft-Thresholding** ke zariye!

---

## 🧭 Preprocessing Notice:
> 📌 *Data Ingestion, Cleaning, Scaling, aur Train-Test Split ke standard steps ke liye refer karein:*  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📐 Section 1: The Lasso Loss Function

$$\text{Loss}_{\text{Lasso}} = \frac{1}{2n} \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 + \alpha \sum_{j=1}^p |w_j|$$

- **Part 1: $\text{RSS}$ (Residual Sum of Squares):** Model ko bolta hai *"Data par acche se fit ho"*.
- **Part 2: $\alpha \sum |w_j|$ ($L_1$ Penalty / Tax):** Model ko bolta hai *"Har feature ke weight ke magnitude par tax lagega"*.
- **$\alpha$ (Alpha / Regularization Strength):**
  - $\alpha = 0 \implies$ OLS Linear Regression (No penalty).
  - $\alpha \to \infty \implies$ Saare weights $w_j = 0$ (Model predicts constant mean).

---

## 🚫 Section 2: Why Normal Calculus (Closed-Form) Fails for Lasso

Ridge Regression mein humne derivative lekar zero ke barabar rakh diya tha:
$$\frac{\partial J}{\partial w} = 0 \implies \hat{w}_{\text{ridge}} = (X^T X + \alpha I)^{-1} X^T y$$

Lekin Lasso mein:
$$\frac{d}{dw} |w| = \begin{cases} +1 & \text{if } w > 0 \\ -1 & \text{if } w < 0 \\ \text{UNDEFINED!} & \text{if } w = 0 \end{cases}$$

$w=0$ par derivative exist hi nahi karta! Isliye **Lasso ka koi closed-form analytical formula $(X^TX)^{-1}$ nahi hota!**

---

## ⚡ Section 3: The Secret Weapon — Coordinate Descent & Soft-Thresholding

Computer Lasso ko **Coordinate Descent** algorithm se solve karta hai:  
Sabhi features ko ek saath update karne ke bajaye, ek time par **sirf ek feature ($w_j$)** ko optimize karta hai jabki baaki sabhi features ko freeze (constant) rakhta hai.

Single feature update ka analytical solution nikalta hai jisko bolte hain **Soft-Thresholding Operator**:

$$\hat{w}_j = S\left(\rho_j, \alpha\right) = \text{sign}(\rho_j) \cdot \max\left(|\rho_j| - \alpha, \, 0\right)$$

jahan $\rho_j$ us feature ka OLS correlation/residual contribution hai.

### 🎯 Soft-Thresholding Logic (Kyu exact ZERO banta hai?):
1. **Case 1: Weak Feature ($|\rho_j| \le \alpha$):**  
   Agar feature ka contribution penalty $\alpha$ se kam hai:  
   $$\hat{w}_j = 0.0$$  
   *(Feature model se permanently gayab!)*
2. **Case 2: Strong Positive Feature ($\rho_j > \alpha$):**  
   $$\hat{w}_j = \rho_j - \alpha$$  
   *(Weight thoda sa shrink ho kar zinda rehta hai).*
3. **Case 3: Strong Negative Feature ($\rho_j < -\alpha$):**  
   $$\hat{w}_j = \rho_j + \alpha$$  
   *(Weight thoda sa shrink ho kar negative zinda rehta hai).*

---

## 📊 Visualizing the Shrinkage Difference

```
Ridge Shrinkage (Linear):          Lasso Shrinkage (Soft-Thresholding):
w_ridge = w_ols / (1 + alpha)       w_lasso = 0  if  |w_ols| <= alpha

      w_ridge                              w_lasso
         |   /                                |     /
         |  /                                 |    /
         | /                                  |   /
---------+---------                  ---------+---+---+---------
        /|                                        |   |
       / |                                       /    |
      /  |                                      /     |
(Never reaches zero)                 (Flattens to EXACT 0 in [-α, +α] zone!)
```

