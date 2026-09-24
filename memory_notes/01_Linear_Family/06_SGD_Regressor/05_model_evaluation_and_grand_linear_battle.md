# ⚔️ SGD Regressor Memory Notes: 05 - The Grand Linear Battle Showdown

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Linear Family ke 5 maharathiyon ka antim yuddh (Grand Arena Battle):  
> 1. **OLS Regression:** (Unpenalized baseline)  
> 2. **Ridge Regression:** ($L_2$ Closed-Form)  
> 3. **Lasso Regression:** ($L_1$ Coordinate Descent)  
> 4. **ElasticNet Regression:** ($L_1 + L_2$ Coordinate Descent)  
> 5. **SGDRegressor:** (Iterative Gradient Descent Champion)  
> Dekhein ki kya iterative gradient descent closed-form models ke barabar accuracy de pata hai ya nahi!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚔️ Step 9: The Grand 5-Way Battle Implementation Code

```python
# ==============================================================================
# ⚔️ STEP 9: THE 5-WAY GRAND LINEAR BATTLE SHOWDOWN
# ==============================================================================
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# 1. Fit All 5 Contenders on Processed Train Data
ols_m = LinearRegression().fit(X_train_final, y_tr)
ridge_m = Ridge(alpha=ridge_alpha, random_state=42).fit(X_train_final, y_tr)
lasso_m = Lasso(alpha=lasso_alpha, max_iter=10000, tol=1e-3, random_state=42).fit(X_train_final, y_tr)
elastic_m = ElasticNet(alpha=elastic_alpha, l1_ratio=elastic_rho, max_iter=10000, tol=1e-3, random_state=42).fit(X_train_final, y_tr)
sgd_m = best_sgd

# 2. Predict on Test Set (Log Scale)
preds_log = {
    'OLS Linear Regression': ols_m.predict(X_test_final),
    'Ridge Regression (L2)': ridge_m.predict(X_test_final),
    'Lasso Regression (L1)': lasso_m.predict(X_test_final),
    'ElasticNet (L1 + L2)':  elastic_m.predict(X_test_final),
    'SGD Regressor (Iterative)': sgd_m.predict(X_test_final)
}

# 3. Invert Log Scale back to Real Business Units (np.expm1)
y_test_real = np.expm1(y_te)

# 4. Compile Metrics Table
models_dict = {
    'OLS Linear Regression': ols_m,
    'Ridge Regression (L2)': ridge_m,
    'Lasso Regression (L1)': lasso_m,
    'ElasticNet (L1 + L2)':  elastic_m,
    'SGD Regressor (Iterative)': sgd_m
}

rows = []
for name, mod in models_dict.items():
    p_log = preds_log[name]
    p_real = np.expm1(p_log)
    
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

print("=" * 95)
print("🏆 THE GRAND 5-WAY LINEAR BATTLE SHOWDOWN")
print("=" * 95)
print(battle_df.to_string(index=False))
print("=" * 95)
```

---

## 💡 Key Architectural Takeaway: Exact vs. Approximate Solution

| Algorithm | Solver Technique | Precision | Scalability to 10M rows |
| :--- | :--- | :--- | :--- |
| **OLS / Ridge** | Closed-Form Analytical ($(\mathbf{X}^T \mathbf{X})^{-1}$) | Exact (100%) | ❌ Crashes RAM (OOM) |
| **Lasso / ElasticNet** | Coordinate Descent | Exact (within tolerance) | ⚠️ Moderate (slow on massive data) |
| **SGDRegressor** | Stochastic Gradient Descent | Approximate ($pprox 99.8\%$ of exact) | 🏆 **Infinite Scalability ($O(1)$ RAM)** |
