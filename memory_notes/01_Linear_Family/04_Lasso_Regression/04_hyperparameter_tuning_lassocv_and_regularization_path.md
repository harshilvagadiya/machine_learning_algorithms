# ✂️ Lasso Regression Memory Notes: 04 - Hyperparameter Tuning (`LassoCV`) & Regularization Path

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Lasso mein $\alpha$ (Alpha) ek kanchi (scissors) ki tarah hai:  
> - Agar kanchi choti hai ($\alpha = 0.0001$), toh wo kisi feature ko nahi kaategi (saare zinda rahenge, OLS jaisa overfitting).  
> - Agar kanchi bohot badi hai ($\alpha = 100$), toh wo sabhi features ko kaat kar zero kar degi (underfitting).  
> - **Sahi kanchi ($\alpha^*$)** dhoondhne ke liye hum Scikit-Learn ka `LassoCV` use karte hain!

---

## 🧭 Preprocessing Notice:
> 📌 *Data Ingestion, Cleaning, Scaling, aur Train-Test Split ke standard steps ke liye refer karein:*  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚙️ Section 1: The Critical Convergence Settings (`max_iter` & `tol`)

Lasso Coordinate Descent iterative algorithm hai. Agar settings sahi nahi hui, toh terminal par yeh red warning chhati hai:
```text
ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations.
```

### 3 Rules to Prevent Convergence Warning:
1. **Feature Scaling (Mandatory):** Agar `StandardScaler` nahi lagaya, toh algorithm kabhi converge nahi hoga!
2. **`max_iter` Badhayein:** Default `1000` hota hai, ise `5000` ya `10000` karein.
3. **`tol` (Tolerance):** Default `1e-4` hota hai, ise `1e-3` ya `1e-4` par set karein.

---

## 🧭 Section 2: Why GCV (`cv=None`) Doesn't Exist for Lasso

- **Ridge Regression** mein Hat Matrix $H = X(X^TX+\alpha I)^{-1}X^T$ ka shortcut tha, isliye `cv=None` 0.1 second mein chal jata tha.
- **Lasso Regression** mein analytical matrix formula nahi hota!  
  Isliye LassoCV ko **K-Fold Cross-Validation (`cv=5`)** hi run karna padta hai.
- Lekin Scikit-Learn ek smart trick use karta hai: **Warm Starts along the Alpha Path!**  
  Wo sabse bade $\alpha$ (jahan saare weights 0 hain) se shuru karta hai aur dheere-dheere $\alpha$ chhota karta hai, jisse training super fast ho jaati hai!

---

## 💻 Copy-Paste Boilerplate: Step 8 LassoCV & Regularization Path

```python
# ==============================================================================
# 🏆 STEP 8: LASSOCV HYPERPARAMETER TUNING & FEATURE ELIMINATION TRACE
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold

# 1. 100 Candidate Alphas (0.0001 to 100)
alphas = np.logspace(-4, 2, 100)

# 2. 5-Fold Cross-Validation on Training Data ONLY (Zero Leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_cv = LassoCV(
    alphas=alphas, 
    cv=kf, 
    max_iter=10000, 
    tol=1e-3, 
    random_state=42
)
lasso_cv.fit(X_train_final, y_tr)

best_alpha = lasso_cv.alpha_

# 3. Fit Best Lasso Model
best_lasso = Lasso(alpha=best_alpha, max_iter=10000, tol=1e-3, random_state=42)
best_lasso.fit(X_train_final, y_tr)

# 4. Generate Regularization Path (Features dropping to zero)
coef_paths = []
for a in alphas:
    m = Lasso(alpha=a, max_iter=5000, tol=1e-3, random_state=42)
    m.fit(X_train_final, y_tr)
    coef_paths.append(m.coef_)
coef_paths = np.array(coef_paths)

# 5. Diagnostic Summary
zero_feats = np.sum(best_lasso.coef_ == 0)
total_feats = len(best_lasso.coef_)
active_feats = total_feats - zero_feats

print("=" * 70)
print("🏆 STEP 8: LASSOCV HYPERPARAMETER TUNING REPORT")
print("=" * 70)
print(f"Optimal L1 Penalty (α*)     : {best_alpha:.5f}")
print(f"Total Input Features        : {total_feats} features")
print(f"Active Features Retained    : {active_feats} features (Non-zero)")
print(f"Useless Features Eliminated : {zero_feats} features ({zero_feats/total_feats*100:.1f}% Sparsity!)")
print(f"Model Intercept             : {best_lasso.intercept_:.4f}")
print("=" * 70)

# 6. Visualizing the Lasso Elimination Path (Regularization Trace)
plt.figure(figsize=(10, 6))
for i in range(coef_paths.shape[1]):
    plt.plot(alphas, coef_paths[:, i], label=all_names[i] if i < 6 else None, alpha=0.8, linewidth=1.5)

plt.xscale('log')
plt.axvline(best_alpha, color='red', linestyle='--', linewidth=2, label=f'Optimal α* = {best_alpha:.5f}')
plt.axhline(0, color='black', linestyle=':', alpha=0.7)
plt.title("Lasso Regularization Trace: Features Dropping to EXACT ZERO", fontsize=13, fontweight='bold', pad=12)
plt.xlabel("L1 Penalty Alpha (log scale) -> Higher Alpha = More Features Eliminated", fontsize=11)
plt.ylabel("Standardized Coefficients (Weights)", fontsize=11)
plt.grid(True, which='both', linestyle=':', alpha=0.6)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()
```

---

## 🎯 How to Read the Regularization Trace Graph:
1. **Left Side ($\alpha$ chhota):** Saare features active hain (dense model, OLS jaisa).
2. **Right Side ($\alpha$ bada):** Ek-ek karke features horizontal line ($0.0$) par aate jaate hain aur permanently mar jaate hain!
3. **Red Dashed Line ($\alpha^*$):** Yeh Cross-Validation ka optimal balance point hai jahan model ne best generalization score achieve kiya hai.

