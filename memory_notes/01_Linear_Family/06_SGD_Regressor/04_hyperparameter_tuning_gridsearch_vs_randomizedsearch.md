# 🏆 SGD Regressor Memory Notes: 04 - Hyperparameter Tuning Strategy

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Ridge, Lasso aur ElasticNet ke paas unke apne dedicated cross-validators the (`RidgeCV`, `LassoCV`, `ElasticNetCV`).  
> Lekin SGDRegressor ke paas koi `SGDRegressorCV` nahi hota!  
> Kyun? Kyunki SGD mein sirf alpha nahi, balki **Loss, Penalty, Learning Rate schedule, aur initial eta0** sabko ek saath tune karna hota hai!  
> Isliye hum **`GridSearchCV` ya `RandomizedSearchCV`** use karte hain!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚙️ Step 8: Production Tuning Code with GridSearchCV

```python
# ==============================================================================
# 🏆 STEP 8: PRODUCTION HYPERPARAMETER TUNING FOR SGDREGRESSOR
# ==============================================================================
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np

# 1. Multi-Dimensional Search Space
param_grid = {
    'loss': ['squared_error', 'huber'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [1e-4, 1e-3, 1e-2, 1e-1],
    'learning_rate': ['adaptive', 'invscaling'],
    'eta0': [0.001, 0.01, 0.05]
}

# 2. Strict 5-Fold Cross-Validation on Processed Training Data ONLY (Zero Leakage)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

sgd_base = SGDRegressor(
    max_iter=5000, 
    tol=1e-3, 
    early_stopping=True, 
    n_iter_no_change=5,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=sgd_base,
    param_grid=param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# 3. Fit GridSearch on Training Matrix
grid_search.fit(X_train_final, y_tr)

best_sgd = grid_search.best_estimator_
best_params = grid_search.best_params_

print("=" * 70)
print("🏆 STEP 8: SGDREGRESSOR HYPERPARAMETER TUNING REPORT")
print("=" * 70)
print(f"Optimal Loss Function        : '{best_params['loss']}'")
print(f"Optimal Penalty Type         : '{best_params['penalty']}'")
print(f"Optimal Regularization Alpha : {best_params['alpha']:.5f}")
print(f"Optimal Learning Rate Policy : '{best_params['learning_rate']}'")
print(f"Optimal Initial Step (eta0)  : {best_params['eta0']}")
print(f"Best 5-Fold Cross-Val R²     : {grid_search.best_score_ * 100:.2f}%")
print(f"Total Iterations to Converge : {best_sgd.n_iter_} epochs")
print("=" * 70)
```

---

## 💡 Practical Rules for Tuning SGDRegressor

1. **Start with `eta0=0.01`:** Agar model diverge hota hai (Loss NaN ho jata hai), toh `eta0=0.001` par drop karein.
2. **Use `loss='huber'` if target has extreme tails:** Real estate, stock prices ya high-variance sales data mein `huber` standard squared error se behtar generalize karta hai.
3. **If dataset is huge (> 100k rows), prefer `RandomizedSearchCV`:** 100 candidate evaluations randomly sample karke $5	imes$ time bacha leta hai.
