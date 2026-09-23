# 🌐 Common Linear Family: 04 - Missing Values & Duplicates Hygiene

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> 1. **Duplicates (Photocopies):** Agar ek hi student ka paper do baar photocopy hokar aa gaya, aur ek copy question paper ban gayi aur doosri answer key, to student cheat karke 100/100 le aayega! Isliye duplicate rows ko pehle hi nikaal kar phenkna padta hai.  
> 2. **Missing Values (Khali Jagah):** Agar kisi ghar mein `PoolQC = NaN` likha hai, to iska matlab yeh nahi ki form kho gaya, iska matlab hai ki **ghar mein swimming pool hai hi nahi!** Agar row delete kar doge to 99% data ud jayega!

---

## 🧭 The Data Hygiene Protocol

```
                        DATA HYGIENE PROTOCOL
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
DUPLICATE ROWS AUDIT                                  MISSING VALUES AUDIT
- df.duplicated().sum()                               - MCAR (Accidental null) -> Median/Mode
- Must drop BEFORE train_test_split                   - MNAR (Structural absence) -> 'None'
- Prevents train-test leakage                         - NEVER drop rows blindly with dropna()!
```

---

## 🛑 The Senior Developer Rules for Missingness

| Scenario | Example | Junior Mistake ❌ | Senior Developer Solution ✅ |
| :--- | :--- | :--- | :--- |
| **Amenity Absence (MNAR)** | `PoolQC`, `Fence`, `FireplaceQu` is null | `df.dropna()` (Poora dataset khatam!) | Fill with `'None'` category (Feature ban gaya: "No Pool") |
| **Continuous Metric Null** | `LotFrontage` (street distance) is null | Fill with `0` (Mean distort ho jata hai) | `SimpleImputer(strategy='median')` fitted on train |
| **Identical Duplicate Row** | Same row appears multiple times | Ignore kar diya | `df.drop_duplicates(inplace=True)` before splitting |

---

## 📋 Copy-Paste Boilerplate: Complete Data Hygiene Audit & Cleaner
*(Isko copy karke kisi bhi notebook ke Step 4 mein paste karo)*

```python
# ==============================================================================
# 🌐 STEP 4: DATA HYGIENE AUDIT (DUPLICATES & MISSINGNESS PROFILER)
# ==============================================================================
import pandas as pd

def audit_and_clean_hygiene(df, drop_duplicates=True):
    """
    1. Audits and cleans duplicate rows.
    2. Profiles missing values by count and percentage.
    3. Categorizes columns with nulls.
    """
    print("=" * 70)
    print("🧹 STEP 4: DATA HYGIENE & PURITY REPORT")
    print("=" * 70)
    
    # 1. Duplicates Audit
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        print(f"⚠️ FOUND {n_dups:,} EXACT DUPLICATE ROWS!")
        if drop_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"✅ Cleaned: {n_dups:,} duplicates removed. New Shape: {df.shape}")
    else:
        print("✅ Zero duplicate rows found. Data integrity is clean.")
    print("-" * 70)

    # 2. Missing Values Profiler
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if len(null_cols) > 0:
        null_report = pd.DataFrame({
            "Missing Count": null_cols,
            "Percentage (%)": (null_cols / len(df) * 100).round(2),
            "Data Type": [df[col].dtype for col in null_cols.index]
        })
        print(f"🚨 FOUND {len(null_cols)} COLUMNS WITH MISSING VALUES:")
        print(null_report.to_string())
    else:
        print("✅ Zero missing values across all features. Dataset is complete.")
    print("=" * 70)
    
    return df

# Usage Example:
# df = audit_and_clean_hygiene(df, drop_duplicates=True)
```

