# ✂️ Algorithm 04: Lasso Regression ($L_1$ Regularization) — Complete Memory Notes Suite

> **Dumb Student Master Summary (Ek Line Mein):**  
> **Ridge Regression** sabhi features ko thoda-thoda chhota karta hai lekin kisi ko jail se riha nahi karta.  
> **Lasso Regression** kanchi (scissors) lekar ghumta hai aur faaltu features ke weights ko **exact 0.0** bana kar **Automatic Feature Selection** kar deta hai!

---

## 🧭 Preprocessing & Universal Pipeline Protocol

> 📌 **CRITICAL RULE (DO NOT REPEAT COMMON STEPS):**  
> Linear Family ke pehle 7 steps (Libraries, Ingestion, Sanitization, Feature Segregation, Missing Values Hygiene, Zero-Leakage Split with Target Guard, Skewness Profiler, aur Master ColumnTransformer) har project mein **identical** hote hain!  
> Unke detailed concept notes, mathematical reasoning aur production boilerplate code ke liye refer karein:  
> ➡️ **[00_Common_Linear_Family Master Suite](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📚 Step-by-Step Lasso Memory Notes Suite

Humne Lasso Regression ke core concepts ko 7 dedicated, crystal-clear guides mein organize kiya hai:

| Guide # | Document Title | What You Will Learn (In Bhai Language) |
| :---: | :--- | :--- |
| **01** | [01_the_intuition_and_why_lasso_is_magic.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/01_the_intuition_and_why_lasso_is_magic.md) | 100 suspects aur detective ki kahani, Dense vs. Sparse models, aur real-world use cases. |
| **02** | [02_the_lasso_math_and_l1_regularization.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/02_the_lasso_math_and_l1_regularization.md) | Absolute weight penalty, $w=0$ par calculus kyu fail hota hai, aur Coordinate Descent / Soft-Thresholding. |
| **03** | [03_geometric_intuition_diamond_vs_circle.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/03_geometric_intuition_diamond_vs_circle.md) | Katora (Circle) vs. Heera (Diamond): Geometric proof ki kyu contours axes ke corners par takrate hain. |
| **04** | [04_hyperparameter_tuning_lassocv_and_regularization_path.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/04_hyperparameter_tuning_lassocv_and_regularization_path.md) | Step 8: `LassoCV` tuning, Convergence settings (`max_iter`, `tol`), aur features ko marte hue dekhne wala Trace graph. |
| **05** | [05_model_evaluation_and_triple_battle.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/05_model_evaluation_and_triple_battle.md) | Step 9: The Triple Battle Showdown (OLS vs. Ridge vs. Lasso), Test $R^2$, Real Dollar RMSE, aur Sparsity metrics. |
| **06** | [06_feature_selection_and_sparse_model_interpretation.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/06_feature_selection_and_sparse_model_interpretation.md) | Step 10: Surviving features vs. Eliminated noise extraction, Value Boosters vs. Reducers horizontal bar chart. |
| **07** | [07_lasso_limitations_and_when_to_use_elasticnet.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/04_Lasso_Regression/07_lasso_limitations_and_when_to_use_elasticnet.md) | Lasso ke 3 sabse bade flaws (Collinear lottery, $p>n$ saturation limit, high bias) aur ElasticNet ki entry. |

---

## 📐 Mathematical Formulation Summary

1. **Lasso Loss Function:**
   $$J(w) = \frac{1}{2n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^p |w_j|$$

2. **Soft-Thresholding Operator (Coordinate Descent Update):**
   $$\hat{w}_j = \text{sign}(\rho_j) \cdot \max(|\rho_j| - \alpha, \, 0)$$

3. **Hyperparameter ($\alpha$):**
   - $\alpha = 0 \implies$ OLS Linear Regression.
   - $\alpha \to \infty \implies$ Sparsity 100% (All $w_j = 0$).

---

## ⚡ 10-Second Interview Flashcard (Jab HR / Tech Lead Puche)

| Question | Junior Answer ❌ | Senior "Bhai" Answer ✅ |
| :--- | :--- | :--- |
| **"Lasso aur Ridge mein main fark kya hai?"** | Dono mein regularization hoti hai. | Ridge ($L_2$) weights ko chhota karta hai par zero nahi karta; Lasso ($L_1$) irrelevant weights ko **exact $0.0$** karke **automatic feature selection (sparse model)** karta hai. |
| **"Lasso ka closed-form formula kyu nahi hota?"** | Formula lamba hota hai isliye. | Absolute value $|w|$ ka derivative $w=0$ par undefined hota hai (sharp kink). Isliye analytical solution nahi hota, ise **Coordinate Descent & Soft-Thresholding** se solve kiya jata hai. |
| **"Lasso kab fail ho sakti hai?"** | Kabhi nahi, best model hai. | Agar group of collinear features ho, toh Lasso unme se kisi ek ko randomly chunta hai aur baaki ko zero kar deta hai (instability). Is case mein **ElasticNet** use karte hain. |

---

## 📋 Master Copy-Paste Boilerplate: End-to-End Lasso Modeling (Steps 8-11)
*(Common Steps 1-7 chalaane ke baad is block ko paste karein)*

```python
# ==============================================================================
# ✂️ MASTER BOILERPLATE: LASSOCV, EVALUATION & FEATURE SELECTION
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso, Ridge, LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# 1. STEP 8: LASSOCV HYPERPARAMETER TUNING
alphas = np.logspace(-4, 2, 100)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, tol=1e-3, random_state=42)
lasso_cv.fit(X_train_final, y_tr)

best_alpha = lasso_cv.alpha_
best_lasso = Lasso(alpha=best_alpha, max_iter=10000, tol=1e-3, random_state=42)
best_lasso.fit(X_train_final, y_tr)

zero_feats = np.sum(best_lasso.coef_ == 0)
total_feats = len(best_lasso.coef_)
print(f"🏆 Best Alpha: {best_alpha:.5f} | Retained: {total_feats - zero_feats}/{total_feats} features")

# 2. STEP 9: TEST SET EVALUATION
lasso_pred_log = best_lasso.predict(X_test_final)
y_test_real = np.expm1(y_te) # Invert if log was used
lasso_pred_real = np.expm1(lasso_pred_log)

test_r2 = r2_score(y_te, lasso_pred_log)
test_rmse = root_mean_squared_error(y_test_real, lasso_pred_real)
print(f"📊 Test R²: {test_r2*100:.2f}% | Test RMSE: ${test_rmse:,.0f}")

# 3. STEP 10: EXTRACT SURVIVORS & PLOT
weights_df = pd.DataFrame({'Feature': all_names, 'Weight': best_lasso.coef_})
survivors = weights_df[weights_df['Weight'] != 0].sort_values(by='Weight', ascending=True)

plt.figure(figsize=(9, max(4, len(survivors) * 0.4)))
colors = ['#27ae60' if w > 0 else '#e74c3c' for w in survivors['Weight']]
plt.barh(survivors['Feature'], survivors['Weight'], color=colors, edgecolor='black')
plt.axvline(0, color='black', linestyle='--')
plt.title(f"Lasso Active Features ({len(survivors)} Survived)", fontweight='bold')
plt.tight_layout()
plt.show()

# 4. STEP 11: FULL PRODUCTION PIPELINE SERIALIZATION
prod_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('lasso', best_lasso)
])
prod_pipeline.fit(X_tr, y_tr)

model_path = Path("production_models/lasso_production_pipeline.joblib")
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(prod_pipeline, model_path)
print(f"💾 Production Artifact Saved: {model_path}")
```
