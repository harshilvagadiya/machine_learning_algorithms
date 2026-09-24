# ✂️ Lasso Regression Memory Notes: 05 - The Triple Battle (OLS vs. Ridge vs. Lasso)

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Linear models ke 3 sabse bade pehelwan ek hi ring mein utarte hain:  
> 1. **OLS Linear Regression:** Raw Power (No constraints).  
> 2. **Ridge Regression ($L_2$):** Disciplined Bodybuilder (Saare muscle chhote aur tight rakhta hai).  
> 3. **Lasso Regression ($L_1$):** Ninja Assassin (Sirf zaroori hathiyar rakhta hai, baaki sab fek deta hai).  
> Kaun jeetega? Yeh table faisla karega!

---

## 🧭 Preprocessing Notice:
> 📌 *Data Ingestion, Cleaning, Scaling, aur Train-Test Split ke standard steps ke liye refer karein:*  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 💻 Copy-Paste Boilerplate: Step 9 The Triple Battle

```python
# ==============================================================================
# ⚔️ STEP 9: THE TRIPLE BATTLE: OLS VS. RIDGE VS. LASSO REGRESSION
# ==============================================================================
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# 1. Fit All 3 Contenders on Processed Train Data
ols_m = LinearRegression().fit(X_train_final, y_tr)
ridge_m = Ridge(alpha=best_ridge_alpha, random_state=42).fit(X_train_final, y_tr)
lasso_m = Lasso(alpha=best_lasso_alpha, max_iter=10000, tol=1e-3, random_state=42).fit(X_train_final, y_tr)

# 2. Predict on Test Set (Log Scale)
ols_pred_log = ols_m.predict(X_test_final)
ridge_pred_log = ridge_m.predict(X_test_final)
lasso_pred_log = lasso_m.predict(X_test_final)

# 3. Invert Log Scale back to Real Dollar / Unit Space (np.expm1)
USE_LOG_TARGET = True
if USE_LOG_TARGET:
    y_test_real = np.expm1(y_te)
    ols_pred_real = np.expm1(ols_pred_log)
    ridge_pred_real = np.expm1(ridge_pred_log)
    lasso_pred_real = np.expm1(lasso_pred_log)
else:
    y_test_real = y_te
    ols_pred_real = ols_pred_log
    ridge_pred_real = ridge_pred_log
    lasso_pred_real = lasso_pred_log

# 4. Compute Metrics for All 3 Models
models = {
    'OLS Linear Regression': (ols_m, ols_pred_log, ols_pred_real),
    'Ridge Regression (L2)': (ridge_m, ridge_pred_log, ridge_pred_real),
    'Lasso Regression (L1)': (lasso_m, lasso_pred_log, lasso_pred_real)
}

rows = []
for name, (mod, p_log, p_real) in models.items():
    tr_r2 = r2_score(y_tr, mod.predict(X_train_final))
    te_r2 = r2_score(y_te, p_log)
    rmse = root_mean_squared_error(y_test_real, p_real)
    mae = mean_absolute_error(y_test_real, p_real)
    n_active = np.sum(mod.coef_ != 0)
    total_f = len(mod.coef_)
    
    rows.append({
        'Model Name': name,
        'Train R²': f"{tr_r2*100:.2f}%",
        'Test R²': f"{te_r2*100:.2f}%",
        'Real RMSE': f"${rmse:,.0f}",
        'Real MAE': f"${mae:,.0f}",
        'Active Features': f"{n_active} / {total_f}",
        'Sparsity (%)': f"{(total_f - n_active) / total_f * 100:.1f}%"
    })

battle_df = pd.DataFrame(rows)

print("=" * 85)
print("🏆 THE TRIPLE BATTLE SHOWDOWN: OLS VS. RIDGE VS. LASSO")
print("=" * 85)
print(battle_df.to_string(index=False))
print("=" * 85)
```

---

## 🎯 Executive Decision Guide (Kaun Sa Model Production mein Jaye?)

| Criterion | Best Model | Why? |
| :--- | :--- | :--- |
| **Max Predictive Power (Accuracy)** | **Ridge ($L_2$)** | Agar features mutually correlated hain, toh Ridge unhe retain karke best possible variance control deta hai. |
| **Feature Selection & Interpretability** | **Lasso ($L_1$)** | Agar client ko lightweight, explainable model chahiye jo 80% faaltu inputs ko eliminate kar sake. |
| **High Dimensionality ($p \gg n$)** | **Lasso ($L_1$)** | OLS fail ho jata hai; Lasso sparseness create karke model ko robust banata hai. |
| **Low Latency Real-Time API** | **Lasso ($L_1$)** | Zeroed-out features ko API request mein calculate karne ki zaroorat hi nahi hoti! |

