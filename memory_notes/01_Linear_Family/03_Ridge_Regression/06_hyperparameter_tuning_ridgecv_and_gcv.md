# 🛡️ Ridge Regression Memory Notes: 06 - Hyperparameter Tuning: RidgeCV & GCV

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho chai mein cheeni (sugar) daalni hai:  
> - Cheeni 0 spoon dali $\to$ Chai feeki lagegi ($\alpha = 0$, OLS overfitting).  
> - Cheeni 10 spoon daal di $\to$ Chai chashni ban jayegi, peene layak nahi rahegi ($\alpha = 10,000$, Underfitting).  
> - **Sahi cheeni (2 spoon)** kaise pata chalegi? Tum sip kar-karke (Cross-Validation) test karoge!  
> Scikit-Learn ka `RidgeCV` wahi automated tea-taster hai jo 1 second mein perfect $\alpha$ nikaal leta hai!

---

## 🧭 The Tuning Roadmap

```
                       HOW TO FIND THE BEST ALPHA?
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
THE SLOW WAY (K-FOLD CV)                             THE SUPER SMART WAY (GCV)
- Fits model K times (e.g. 5 or 10 times)           - Generalized Cross-Validation
- Very slow on large feature matrices (333 cols)    - 1-Step mathematical shortcut via Hat Matrix
- Ridge(cv=5)                                       - RidgeCV(cv=None) -> Runs in < 1 second!
```

---

## 🔍 Section 1: The Search Grid (`np.logspace` vs. `np.linspace`)

Alpha ($\alpha$) koi aam number nahi hai, yeh **orders of magnitude (powers of 10)** par kaam karta hai:
- $\alpha = 0.01$ par model lagbhag OLS hota hai.
- $\alpha = 1.0$ par halka sa shrinkage shuru hota hai.
- $\alpha = 10.0$ par strong stability aati hai.
- $\alpha = 100.0$ par weights bohot chhotey ho jaate hain.

Isliye kabhi bhi `np.linspace(1, 100, 10)` mat use karo!  
Hamesha **`np.logspace`** use karo:

```python
import numpy as np

# 50 candidate values from 10^-2 (0.01) to 10^3 (1000.0)
alphas = np.logspace(-2, 3, 50)
```

---

## ⚡ Section 2: Generalized Cross-Validation (GCV) - The Mathematical Magic

Normal Leave-One-Out Cross-Validation (LOOCV) mein agar $n = 1,460$ rows hain:
- Model ko **1,460 baar train** karna padega! Computer hang ho jayega!

Lekin Ridge Regression ke paas ek secret mathematical superpower hai jisko kehte hain **Generalized Cross-Validation (GCV)**:

### The Hat Matrix Shortcut:
$$\hat{y} = H y \quad \text{where } H = X(X^T X + \alpha I)^{-1} X^T$$

GCV bina model ko baar-baar fit kiye, sirf ek single matrix inversion se LOOCV error nikaal leta hai:

$$\text{LOOCV Error}_i = \frac{y_i - \hat{y}_i}{1 - H_{ii}}$$

Aur GCV formula:
$$\text{GCV}(\alpha) = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{1 - \frac{\text{Tr}(H)}{n}} \right)^2$$

> 🚀 **Business Benefit:**  
> Jo hyperparameter tuning normal models mein 5 minute leti hai, `RidgeCV` usko **0.5 second** mein calculate kar leta hai!

---

## 💻 Python Implementation: Production RidgeCV

```python
import numpy as np
from sklearn.linear_model import RidgeCV

# 1. 50 Candidate Alphas
alphas = np.logspace(-2, 3, 50)

# 2. cv=None automatically activates efficient Generalized Cross-Validation (GCV)!
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=None)

# 3. Fit on Scaled Training Matrix
ridge_cv.fit(X_train_scaled, y_train)

# 4. Extract the Best Alpha
best_alpha = ridge_cv.alpha_
print(f"🏆 Best Alpha Selected by GCV: {best_alpha:.2f}")
# Hamare Ames Housing project mein optimal alpha aaya: 11.51!
```

---

## ❓ Dumb Student FAQs

### Q: Agar best alpha search range ke bilkul border (jaise 0.01 ya 1000) par aaye to kya karein?
**Danger Sign!** Agar best alpha $0.01$ aaya, iska matlab aapko aur chhote numbers ($10^{-4}$) check karne chahiye the. Agar $1000$ aaya, to aapko aur bade numbers ($10^5$) check karne chahiye the. Best alpha hamesha search grid ke **beech mein** aana chahiye (jaise hamara $11.51$ aaya).

---

## 📋 Copy-Paste Boilerplate: Automated RidgeCV Alpha Tuner & CV Curve Plotter
*(Isko copy karo, yeh 1 second mein best alpha nikaal kar CV Error ka curve bhi plot kar dega)*

```python
# ==============================================================================
# COPY-PASTE SNIPPET: FAST RIDGECV TUNER & LOSS CURVE PLOTTER
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

def tune_ridge_hyperparameters(X_tr_scaled, y_tr, low_exp=-2, high_exp=3, n_candidates=50):
    """
    Automated Generalized Cross-Validation using RidgeCV.
    Finds best alpha across logspace(low_exp, high_exp, n_candidates) and plots error curve.
    """
    # 1. Generate logarithmic candidate alphas
    alphas = np.logspace(low_exp, high_exp, n_candidates)
    
    # 2. Fit RidgeCV with GCV and store CV loss
    ridge_cv = RidgeCV(
        alphas=alphas, 
        scoring='neg_mean_squared_error', 
        store_cv_values=True,
        cv=None # Activates instantaneous Generalized Cross-Validation!
    )
    ridge_cv.fit(X_tr_scaled, y_tr)
    
    best_alpha = ridge_cv.alpha_
    
    # 3. Compute mean MSE across cross-validation values
    # cv_values_ shape: (n_samples, n_alphas)
    mean_cv_mse = np.mean(ridge_cv.cv_values_, axis=0)
    
    print("=" * 70)
    print("🏆 RIDGECV OPTIMAL HYPERPARAMETER TUNING REPORT")
    print("=" * 70)
    print(f"Alpha Candidate Range     : {alphas[0]:.4f} to {alphas[-1]:.1f} ({n_candidates} candidates)")
    print(f"Optimal L2 Penalty (α)   : {best_alpha:.4f}")
    print(f"Minimum CV MSE (Log-loss): {np.min(mean_cv_mse):.6f}")
    
    # Check if border value
    if np.isclose(best_alpha, alphas[0]) or np.isclose(best_alpha, alphas[-1]):
        print("⚠️ WARNING: Best alpha hit search grid boundary! Expand low_exp or high_exp.")
    else:
        print("✅ Alpha found safely within internal grid bounds (Optimal Sweet Spot).")
    print("=" * 70)
    
    # 4. Plot Cross-Validation Error Curve
    plt.figure(figsize=(9, 4.5))
    plt.plot(alphas, mean_cv_mse, color='darkblue', linewidth=2, label='Mean GCV Error')
    plt.axvline(best_alpha, color='crimson', linestyle='--', linewidth=1.5, label=f'Best α = {best_alpha:.2f}')
    plt.scatter([best_alpha], [np.min(mean_cv_mse)], color='crimson', s=80, zorder=5)
    plt.xscale('log')
    plt.xlabel('Alpha (Log Scale)', fontsize=11, fontweight='bold')
    plt.ylabel('Generalized CV MSE Error', fontsize=11, fontweight='bold')
    plt.title('RidgeCV Optimization Curve: Finding the Minimum Loss Sweet Spot', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return ridge_cv, best_alpha

# Usage:
# tuned_ridge, best_alpha = tune_ridge_hyperparameters(X_train_scaled, y_train, low_exp=-2, high_exp=3)
```


