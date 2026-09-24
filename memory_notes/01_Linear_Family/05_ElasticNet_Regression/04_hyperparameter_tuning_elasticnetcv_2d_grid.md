# 🏆 ElasticNet Memory Notes: 04 - Hyperparameter Tuning via ElasticNetCV (2D Grid)

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Ridge aur Lasso mein sirf **1 knob** tune karna padta tha (`alpha`).  
> Lekin ElasticNet mein **2 knobs** hain: `alpha` (kitni penalty lagani hai) aur `l1_ratio` (Lasso aur Ridge ka mix kitna rakhna hai).  
> Scikit-Learn ka `ElasticNetCV` function automatically **2D Search Matrix** banata hai aur cross-validation karke sabse best `(alpha*, l1_ratio*)` pair dhoondh nikalta hai!

---

## 🧭 Preprocessing Notice (Follow Common Steps)

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚙️ Step 8: Production Code for ElasticNetCV

```python
# ==============================================================================
# 🏆 STEP 8: ELASTICNETCV 2D HYPERPARAMETER TUNING REPORT
# ==============================================================================
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import KFold

# 1. 2D Search Space Define Karein
# l1_ratio: 0.1 (mostly Ridge) se 0.99 (mostly Lasso) tak
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

# alphas: 50 se 100 logarithmic points
alphas = np.logspace(-4, 1, 100)

# 2. Strict 5-Fold Cross-Validation on Training Data ONLY (Zero Leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

elastic_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=alphas,
    cv=kf,
    max_iter=10000,
    tol=1e-3,
    random_state=42,
    n_jobs=-1
)
elastic_cv.fit(X_train_final, y_tr)

# 3. Extract Best Optimal Parameters
best_alpha = elastic_cv.alpha_
best_l1_ratio = elastic_cv.l1_ratio_

# 4. Fit Final Best ElasticNet Model
best_elastic = ElasticNet(
    alpha=best_alpha, 
    l1_ratio=best_l1_ratio, 
    max_iter=10000, 
    tol=1e-3, 
    random_state=42
)
best_elastic.fit(X_train_final, y_tr)

# 5. Diagnostic Sparsity Report
zero_feats = np.sum(best_elastic.coef_ == 0)
total_feats = len(best_elastic.coef_)
active_feats = total_feats - zero_feats

print("=" * 70)
print("🏆 STEP 8: ELASTICNETCV 2D HYPERPARAMETER TUNING REPORT")
print("=" * 70)
print(f"Optimal Alpha (Total Penalty) : {best_alpha:.5f}")
print(f"Optimal L1 Ratio (L1 vs L2)   : {best_l1_ratio:.2f}")
print(f"   • L1 (Lasso Sparsity Part) : {best_l1_ratio * 100:.0f}%")
print(f"   • L2 (Ridge Grouping Part) : {(1 - best_l1_ratio) * 100:.0f}%")
print(f"Total Input Features          : {total_feats} features")
print(f"Active Features Retained      : {active_feats} features (Non-zero)")
print(f"Useless Features Eliminated   : {zero_feats} features ({zero_feats/total_feats*100:.1f}% Sparsity!)")
print(f"Model Intercept (log scale)   : {best_elastic.intercept_:.4f}")
print("=" * 70)
```

---

## 💡 Practical Rules of Thumb for 2D Grid Tuning

1. **Avoid `l1_ratio = [0.0]` or `[1.0]` inside `ElasticNetCV`:**
   - Scikit-Learn `ElasticNetCV` coordinate descent optimizer is designed for $0 < ho < 1$.
   - Pure $ho = 1$ ke liye `LassoCV` use karein; pure $ho = 0$ ke liye `RidgeCV` use karein.
   - ElasticNetCV ke liye best range: `[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]`.
2. **Increase `max_iter` and adjust `tol`:**
   - Jab $ho 	o 1$ hota hai, coordinate descent slow converge hota hai. Isliye `max_iter=10000` aur `tol=1e-3` standard enterprise settings hain.
3. **Always use Standardized Features:**
   - Dono $L_1$ aur $L_2$ penalties scale-sensitive hain. Step 7 ke `StandardScaler` ke bina ElasticNet tuning bekar ho jayegi!
