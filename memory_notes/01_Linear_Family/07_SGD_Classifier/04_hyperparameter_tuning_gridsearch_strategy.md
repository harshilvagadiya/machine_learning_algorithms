# 🏆 SGD Classifier Memory Notes: 04 - Hyperparameter Tuning Strategy via GridSearchCV

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> `SGDClassifier` mein 5 critical levers hote hain jinhe ek saath tune karna zaroori hota hai:  
> 1. `loss`: Kaisa classifier chahiye (`log_loss` vs `hinge` vs `modified_huber`).  
> 2. `penalty`: Konsi regularization chahiye (`l2` vs `l1` vs `elasticnet`).  
> 3. `alpha`: Kitni penalty lagani hai (`0.0001` to `0.01`).  
> 4. `learning_rate`: Kadam kaise decay honge (`adaptive` vs `optimal`).  
> 5. `class_weight`: Agar fraud sirf 1% hai, toh `class_weight='balanced'` lagakar model ko bolna ki minority class par 100x dhyan de!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚙️ Step 8: Production Code for SGDClassifier Tuning

```python
# ==============================================================================
# 🏆 STEP 8: PRODUCTION HYPERPARAMETER TUNING FOR SGDCLASSIFIER
# ==============================================================================
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np

# 1. Multi-Parameter Search Grid
param_grid = {
    'loss': ['log_loss', 'hinge', 'modified_huber'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate': ['adaptive', 'optimal'],
    'eta0': [0.01, 0.05],
    'class_weight': ['balanced', None]
}

# 2. Strict Stratified 5-Fold Cross-Validation on Processed Training Data ONLY
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

sgd_base = SGDClassifier(
    max_iter=5000, 
    tol=1e-3, 
    early_stopping=True, 
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

# 3. Optimize for ROC-AUC (Handles Class Imbalance Elegantly)
grid_search = GridSearchCV(
    estimator=sgd_base,
    param_grid=param_grid,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_tr)

best_sgd = grid_search.best_estimator_
best_params = grid_search.best_params_

print("=" * 70)
print("🏆 STEP 8: SGDCLASSIFIER HYPERPARAMETER TUNING REPORT")
print("=" * 70)
print(f"Optimal Loss Function        : '{best_params['loss']}'")
print(f"Optimal Penalty Type         : '{best_params['penalty']}'")
print(f"Optimal Regularization Alpha : {best_params['alpha']:.5f}")
print(f"Optimal Learning Rate Policy : '{best_params['learning_rate']}'")
print(f"Optimal Class Weight         : {best_params['class_weight']}")
print(f"Best 5-Fold Cross-Val ROC-AUC: {grid_search.best_score_ * 100:.2f}%")
print(f"Epochs to Converge           : {best_sgd.n_iter_} iterations")
print("=" * 70)
```

---

## 💡 Practical Rules for Tuning SGDClassifier

1. **Use `class_weight='balanced'` when Target is Skewed:** Fraud, churn ya cancer data mein `class_weight='balanced'` minority class ka gradient inverse frequency se scale kar deta hai, jisse recall $2	imes$ boost ho jati hai!
2. **Scoring Metric selection in GridSearchCV:**
   - Balanced binary target $\implies$ `scoring='roc_auc'` or `'accuracy'`.
   - Imbalanced target $\implies$ `scoring='f1'` or `'roc_auc'`.
3. **If you need `predict_proba()`:** Grid mein sirf `['log_loss', 'modified_huber']` rakhein. Hinge loss select hui toh `predict_proba` call karne par exception aayega.
