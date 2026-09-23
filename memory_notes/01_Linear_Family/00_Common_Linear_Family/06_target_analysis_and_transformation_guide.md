# 🌐 Common Linear Family: 06 - Target Analysis & Transformation Guide

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Target variable ($y$) hamari movie ka Hero hai. Agar Hero hi bimaar ho, to poori movie flop ho jayegi!  
> 1. **Regression Mein:** Agar gharon ke price right side mein bohot lambi puchh (Skewness) bana rahe hain, to Linear model chilla padega. Humein `np.log1p()` ka injection dekar usko seedha (bell curve) banana padta hai!  
> 2. **Classification Mein:** Agar 100 mein se 95 log healthy hain aur sirf 5 bimaar hain, to model aalsi student ban kar sabko "healthy" bol dega aur 95% marks le aayega bina kuch seekhe! Isko kehte hain **Baseline Accuracy Trap**!

---

## 🧭 The Target Diagnostics Workflow

```
                        TARGET VARIABLE DIAGNOSTICS
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         ▼                                                       ▼
REGRESSION TARGET (CONTINUOUS)                         CLASSIFICATION TARGET (BINARY/MULTI)
- Check Skewness (Ideal: -0.5 to +0.5)                 - Value Counts & Class Ratio (%)
- Right-skewed (Skew > 1.0) -> np.log1p()              - Class Imbalance Check (> 80:20?)
- Real valuation at inference -> np.expm1()            - Zero-Rule Majority Baseline Accuracy
```

---

## 📐 The Regression Skewness Cure: Log-Transformation

Jab target variable right-skewed hota hai (e.g. Real Estate prices, Salaries):
- 90% ghar ₹50 Lakh ke aas-paas hote hain, lekin 2-3 mahal ₹50 Crore ke hote hain.
- Yeh bade numbers Linear Regression ki line ko upar khinch lete hain.

$$\text{Skewness} = \frac{\sum (y_i - \bar{y})^3}{(n-1) s^3}$$

### The Solution:
$$y_{\text{norm}} = \ln(1 + y) \implies \text{Code: } \texttt{np.log1p(y)}$$

Jab model prediction de, to wapas real rupees mein laane ke liye:
$$\hat{y}_{\text{real}} = e^{\hat{y}} - 1 \implies \text{Code: } \texttt{np.expm1(y\_pred)}$$

---

## ⚖️ The Classification Trap: Class Imbalance

Agar data mein:
- Class 0 (Spam nahi hai): 90%
- Class 1 (Spam hai): 10%

Agar tumhara model ek bhi line ka math na lagaye aur aankh band karke bole: *"Bhai, sab Class 0 hain!"*, to model ki **Accuracy 90%** hogi!  
Lekin wo ek bhi Spam pakad nahi paya!  
Isliye classification mein sirf **Accuracy** nahi, balki **Precision, Recall, F1-Score, aur ROC-AUC** check karna padta hai!

---

## 📋 Copy-Paste Boilerplate: Automated Target Profiler & Transformer
*(Isko copy karke kisi bhi notebook ke Target EDA cell mein paste karo)*

```python
# ==============================================================================
# 🌐 STEP 6: TARGET ANALYSIS & TRANSFORMATION PROFILER
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def profile_target_variable(df, target_col, is_classification=False):
    """
    Analyzes target distribution:
    - Regression: Checks skewness, plots raw vs log-transformed bell curves.
    - Classification: Checks class ratios, plots count plot, computes zero-rule baseline accuracy.
    """
    y = df[target_col]
    print("=" * 70)
    print(f"🎯 STEP 6: TARGET VARIABLE PROFILE ({target_col})")
    print("=" * 70)
    
    if not is_classification:
        # --- REGRESSION TARGET ANALYSIS ---
        raw_skew = y.skew()
        y_log = np.log1p(y)
        log_skew = y_log.skew()
        
        print(f"Target Type             : Continuous Regression Variable")
        print(f"Sample Count            : {len(y):,}")
        print(f"Mean (Average)          : {y.mean():,.2f}")
        print(f"Median (50th percentile): {y.median():,.2f}")
        print(f"Standard Deviation      : {y.std():,.2f}")
        print(f"Min / Max               : {y.min():,.2f} / {y.max():,.2f}")
        print(f"Raw Skewness            : {raw_skew:.2f} ({'⚠️ Highly Skewed! Log-transform recommended' if abs(raw_skew) > 0.75 else '✅ Normal'})")
        print(f"Log1p Skewness          : {log_skew:.2f} (Gaussian-like bell curve achieved!)")
        print("=" * 70)
        
        # Dual-panel visual distribution
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.histplot(y, kde=True, ax=axes[0], color='crimson')
        axes[0].set_title(f"Raw Target Distribution (Skew: {raw_skew:.2f})", fontweight='bold')
        
        sns.histplot(y_log, kde=True, ax=axes[1], color='teal')
        axes[1].set_title(f"Log1p Transformed Distribution (Skew: {log_skew:.2f})", fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    else:
        # --- CLASSIFICATION TARGET ANALYSIS ---
        counts = y.value_counts()
        percentages = y.value_counts(normalize=True) * 100
        majority_class_pct = percentages.iloc[0]
        
        print(f"Target Type             : Discrete Classification Variable")
        print(f"Class Breakdown:")
        for cls, count in counts.items():
            print(f"  • Class '{cls}' : {count:,} records ({percentages[cls]:.2f}%)")
        print("-" * 70)
        print(f"🚨 ZERO-RULE MAJORITY BASELINE ACCURACY: {majority_class_pct:.2f}%")
        print(f"   (Any ML model must achieve significantly higher than {majority_class_pct:.2f}% to be useful!)")
        print("=" * 70)
        
        plt.figure(figsize=(6, 4))
        sns.countplot(x=y, palette='Set2')
        plt.title(f"Class Distribution: {target_col}", fontweight='bold')
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

# Usage:
# profile_target_variable(df, target_col="SalePrice", is_classification=False)
# profile_target_variable(df, target_col="Churn", is_classification=True)
```

