# 🌐 Common Linear Family: 05 - Train-Test Split & Zero Leakage Protocol

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho exam se pehle kisi ne question paper leak kar diya aur student ne answer rat liye. Exam mein uske 100/100 marks aa gaye, lekin asal mein usko kuch nahi aata!  
> Machine Learning mein agar tumne **Split karne se pehle** `StandardScaler` ya `SimpleImputer` fit kar diya, to Test set ka information Train set mein ghus jata hai (Data Leakage)!  
> Model train mein 99% accuracy dikhayega aur production mein jaate hi crash ho jayega!

---

## 🧭 The Strict Splitting Rule

```
                                  CLEAN DATASET
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
TRAIN-TEST SPLIT FIRST (80:20)                             ZERO PREPROCESSING BEFORE SPLIT!
- X_train, X_test, y_train, y_test                         - Do NOT fit scaler before split
- Regression: random_state=42                              - Do NOT fit imputer before split
- Classification: stratify=y (Mandatory!)                  - Keep test set 100% blind and unseen!
```

---

## 🎯 Target Preprocessing: Regression vs. Classification

### 1. Regression (Continuous Target): The Log-Transform Rule
Linear models normal distribution (Gaussian Bell curve) pasand karte hain.  
Agar target variable (jaise `SalePrice` ya `Salary`) right-skewed hai (skewness > 1.0):
- **Apply:** `y_train_log = np.log1p(y_train)`
- **Inference time:** `y_real = np.expm1(y_pred_log)`

### 2. Classification (Discrete Target): The Stratification Rule
Agar target binary classification hai (`0` vs `1`):
### 3. The Supervised Learning Golden Rule: TARGET CANNOT CONTAIN NaN!
> 🚨 **Critical Rule (Khatre ki Ghanti):**  
> Features ($X$) mein missing values ho sakti hain kyunki `SimpleImputer` unhe fill kar deta hai.  
> Lekin **Target ($y$) mein ek bhi `NaN` allowed nahi hota!**  
> Scikit-Learn ka `check_X_y` validator finite labels check karta hai. Agar $y$ mein `NaN` chala gaya, to model fit hote hi crash ho jayega:  
> `ValueError: Input y contains NaN.`  
> Isliye split karne se pehle target column ke saare `NaN` rows ko automatically drop karna **MANDATORY** hai!

---

## 📋 Copy-Paste Boilerplate: Zero-Leakage Train-Test Splitter (With Target Guard)
*(Isko copy karke kisi bhi notebook ke Step 5 mein paste karo)*

```python
# ==============================================================================
# 🌐 STEP 5: ENTERPRISE TRAIN-TEST SPLITTER (ZERO DATA LEAKAGE + TARGET GUARD)
# ==============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def perform_zero_leakage_split(df, target_col, drop_cols=None, is_classification=False, apply_log_target=False, test_size=0.20, random_state=42):
    """
    Executes a strict zero-leakage Train-Test split.
    - Safety: Automatically filters out rows where target_col is NaN (Supervised Learning Golden Rule).
    - If classification: Uses stratify=y.
    - If regression & apply_log_target: Applies np.log1p transformation to y.
    """
    if drop_cols is None:
        drop_cols = []
        
    # 0. TARGET INTEGRITY GUARD: Supervised learning requires valid ground-truth labels!
    if df[target_col].isnull().any():
        n_dropped = df[target_col].isnull().sum()
        print(f"⚠️ Target Safety Guard: Dropping {n_dropped:,} rows where target '{target_col}' is NaN!")
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    
    # 1. Target Log-Transformation for Right-Skewed Regression
    if not is_classification and apply_log_target:
        skew_before = y.skew()
        y = np.log1p(y)
        skew_after = y.skew()
        print(f"📈 Target Skewness Normalization: {skew_before:.2f} -> {skew_after:.2f} (Bell curve achieved!)")

    # 2. Perform Split
    stratify_arg = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    
    print("=" * 70)
    print("📦 STEP 5: TRAIN-TEST SPLIT SUMMARY (ZERO LEAKAGE)")
    print("=" * 70)
    print(f"Total Records          : {len(df):,}")
    print(f"Training Matrix (X_tr) : {X_train.shape[0]:,} rows × {X_train.shape[1]} features ({(1-test_size)*100:.0f}%)")
    print(f"Testing Matrix  (X_te) : {X_test.shape[0]:,} rows × {X_test.shape[1]} features ({test_size*100:.0f}%)")
    print(f"Target Missing in y_tr : {y_train.isnull().sum()} nulls (Guaranteed Clean!)")
    if is_classification:
        tr_pos = (y_train == 1).mean() * 100
        te_pos = (y_test == 1).mean() * 100
        print(f"Stratification Balance : Train Class 1 = {tr_pos:.2f}% | Test Class 1 = {te_pos:.2f}% (Identical!)")
    print("=" * 70)
    
    return X_train, X_test, y_train, y_test

# Usage Example:
# X_tr, X_te, y_tr, y_te = perform_zero_leakage_split(df, target_col='SalePrice', drop_cols=['Id'], apply_log_target=True)
```


