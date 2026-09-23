# 🎯 Classification Memory Notes: 03 - Missing Values Audit & Imputation Strategy

> **The Senior Developer's Universal Rule for ANY ML Project:**
> Yeh Memory Note kisi ek project (jaise Heart Disease ya Spam) ke liye nahi hai. Yeh tumhari **Universal Production Playbook** hai jise tum kal ko Credit Card Fraud, Customer Churn, E-Commerce, Healthcare, ya IoT ke kisi bhi dataset par bina ek line change kiye copy-paste kar sako!
> 
> **The 2 Universal Laws:**
> 1. **Target ($y$) missing hai?** $\implies$ **Always drop row!** (`df.dropna(subset=[target_col])`). Unlabelled data se supervised model kabhi nahi seekh sakta.
> 2. **Features ($X$) missing hain?** $\implies$ **Never drop or fill blindly!** Missingness khud ek feature ho sakti hai (MNAR). Aur imputation **sirf aur sirf Train data** par fit hoga (Zero Data Leakage Rule).

---

## 🧭 The Universal Missing Values Decision Tree (For Any Dataset)

```
                            MISSING VALUE DETECTED IN ANY DATASET
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
              TARGET (y) MEIN HAI?                           FEATURE (X) MEIN HAI?
                      │                                               │
                      ▼                                               ▼
             🔥 ALWAYS DROP ROW!                              KITNA PERCENT MISSING HAI?
             `df.dropna(subset=[target_col])`                         │
                                                    ┌─────────────────┼─────────────────┐
                                                    ▼                 ▼                 ▼
                                               LOW (< 5%)     MODERATE (5%-30%)   HIGH (> 50%-70%)
                                                    │                 │                 │
                                        ┌───────────┴──────────┐      │                 ▼
                                        ▼                      ▼      │           BUSINESS CONTEXT:
                                   BIG DATA               SMALL DATA  │           1. High Predictive Signal
                                 (Drop Rows)            (Impute Value)│              -> Keep + 'Missing' category
                                 `df.dropna()`                        │              -> Or MissingIndicator()
                                                                      │           2. Low Value / No Signal
                                                                      │              -> Drop Column!
                                                                      │
                                                    ┌─────────────────┴─────────────────┐
                                                    ▼                                   ▼
                                            NUMERICAL FEATURES                  CATEGORICAL FEATURES
                                                    │                                   │
                                          ┌─────────┴─────────┐               ┌─────────┴─────────┐
                                          ▼                   ▼               ▼                   ▼
                                     SKEWED / OUTLIERS     BELL CURVE      LOW MISSING (<10%)  HIGH MISSING (>20%)
                                     Median Imputer        Mean Imputer    Mode (Most Frequent) Dedicated 'Missing'
                                     (Outlier proof)       (Normal data)   fill_value=mode     category
```

---

## 🔬 Section 1: The 3 Scientific Missingness Mechanisms (Universal Concepts)

Har industry ke dataset mein missing values 3 scientific reasons se aati hain:

### 1. MCAR (Missing Completely At Random) — Pure Accident / Random
* **Matlab:** Data gayab hone ka kisi doosre feature ya outcome se koi relation nahi hai.
* **Real-world Examples:**
  * IoT / Sensors: Battery achanak low ho gayi ya Wi-Fi glitch hua.
  * Customer Form: User ne optional phone number field chhod diya.
* **Universal Action:** Agar MCAR rows $< 5\%$ hain, toh unhe drop karna safe hai; dataset mein koi bias create nahi hota. Agar data chhota hai, toh simple Mean/Median/Mode imputation best hai.

### 2. MAR (Missing At Random) — Systematic but Explainable by Other Features
* **Matlab:** Missingness kisi doosre feature par depend karti hai jo humare paas already dataset mein maujood hai.
* **Real-world Examples:**
  * E-Commerce / Surveys: Younger generation annual income disclose nahi karti, lekin age group pata hai.
  * Banking: Salaried employees ke paas business tax returns missing hote hain.
* **Universal Action:** Doosre features ke relation ka use karke impute karo jaise **KNNImputer** ya **IterativeImputer (MICE)**.

### 3. MNAR (Missing Not At Random) — The Missingness ITSELF is a Feature!
* **Matlab:** Data isliye missing hai kyunki value aisi thi jo record nahi ho saki!
* **Real-world Examples:**
  * Credit Risk / Loans: Jis customer ke paas pichle 2 saal ka credit score missing hai, iska matlab usne kabhi loan nahi liya (New to Credit)!
  * Medical / Hospitals: Jis patient ka expensive scan missing hai, doctor ne usko low-risk samjha isliye test order nahi kiya!
* **Universal Action:** **KABHI BHI IS SIGNAL KO MITTAO MAT!** Agar tumne median se bhar diya toh model ko pata hi nahi chalega ki test hua tha ya nahi. Hamesha **`MissingIndicator`** lagao ya Categorical feature mein `'Missing'` naam ki nayi category banao!

---

## 🟢 Section 2: Universal Plug & Play Audit Function (For Any Future Project)

Is function ko kisi bhi naye project mein as-it-is copy-paste karo. Yeh dynamically poore dataset ka audit nikaal kar deta hai:

```python
import pandas as pd
import numpy as np

def universal_missing_audit(df, target_col=None):
    """
    Universally audits missing values for ANY dataset.
    Works for any number of rows and columns.
    """
    total_rows = len(df)
    total_nulls = df.isnull().sum()
    null_pct = (total_nulls / total_rows) * 100
    
    audit_table = pd.DataFrame({
        'Feature': df.columns,
        'Missing_Count': total_nulls,
        'Percentage (%)': null_pct.round(2),
        'Dtype': df.dtypes,
        'Cardinality': [df[col].nunique(dropna=True) for col in df.columns]
    }).reset_index(drop=True)
    
    # Filter only columns having missing values
    missing_only = audit_table[audit_table['Missing_Count'] > 0].sort_values(
        by='Missing_Count', ascending=False
    ).reset_index(drop=True)
    
    print("=" * 70)
    print("📋 UNIVERSAL MISSING VALUES AUDIT REPORT")
    print("=" * 70)
    print(f"Total Rows:     {total_rows:,}")
    print(f"Total Columns:  {df.shape[1]}")
    print(f"Null Columns:   {len(missing_only)} / {df.shape[1]} ({(len(missing_only)/df.shape[1])*100:.1f}%)")
    
    if target_col and target_col in df.columns:
        target_nulls = df[target_col].isnull().sum()
        status = "✅ 100% Clean (0 nulls)" if target_nulls == 0 else f"⚠️ ALERT: {target_nulls} nulls found! Must drop rows."
        print(f"Target Column:  '{target_col}' -> {status}")
    print("=" * 70)
    
    if len(missing_only) > 0:
        print(missing_only.to_string(index=False))
    else:
        print("🎉 Congratulations! This dataset has ZERO missing values!")
    print("=" * 70)
    
    return missing_only

# How to use in ANY project:
# missing_df = universal_missing_audit(df, target_col='your_target_name')
```

---

## 🟡 Section 3: All 5 Universal Scenarios & Standard Operating Procedures (SOP)

---

### 📌 SOP 1: Target Column Missing ($y$)
* **Condition:** `df[target_col].isnull().sum() > 0`
* **Rule:** Supervised Learning mein target label ke bina row ka koi fayda nahi hota.
* **Code:**
  ```python
  # Always run before anything else:
  df.dropna(subset=[target_col], inplace=True)
  ```

---

### 📌 SOP 2: Extreme Missingness Columns ($> 70\% - 80\%$)
* **Condition:** Kisi feature mein 80% se zyaada data missing hai.
* **Decision:**
  * Agar domain mein koi special signal nahi hai (jaise user comments, optional profile bio) $\implies$ **Drop Column**.
  * Agar missing hona signal hai $\implies$ **Convert to Binary Flag** (`df['had_feature'] = df[col].notnull().astype(int)`).
* **Code:**
  ```python
  def drop_heavy_missing_cols(df, threshold=0.70, exclude_cols=None):
      exclude = exclude_cols or []
      missing_pct = df.isnull().mean()
      drop_cols = [c for c in df.columns if missing_pct[c] > threshold and c not in exclude]
      print(f"Dropping {len(drop_cols)} columns with > {threshold*100}% missing: {drop_cols}")
      return df.drop(columns=drop_cols), drop_cols
  ```

---

### 📌 SOP 3: Numerical Features Missing ($< 50\%$)
* **Condition:** Continuous numerical columns (Income, Age, Transaction Amount, Cholesterol, Blood Pressure).
* **Rule:**
  * Check Skewness (`df[col].skew()`).
  * Agar Skewed / Outliers hain $\implies$ **Median** (`SimpleImputer(strategy='median')`). Median outliers se immune hota hai!
  * Agar Perfect Bell Curve hai $\implies$ **Mean** (`SimpleImputer(strategy='mean')`).
  * *Industry Default:* Hamesha **Median** ko default rakho kyunki real-world datasets mein 95% features skewed hote hain.

---

### 📌 SOP 4: Categorical Features Missing
* **Scenario 4A: Low Missingness ($< 10\%$)**
  * Dominant class exist karti hai $\implies$ **Mode (Most Frequent)**:
    ```python
    SimpleImputer(strategy='most_frequent')
    ```
* **Scenario 4B: High Missingness ($> 20\%$)**
  * Mode daalna distribution ko skew kar dega!
  * **Dedicated Category** banao:
    ```python
    SimpleImputer(strategy='constant', fill_value='Missing')
    ```
    One-Hot Encoding ke baad model ko `col_Missing` feature milega aur woh seekh sakega ki missing hona target ke kitna correlated hai.

---

### 📌 SOP 5: Capturing Missingness as a Feature (`MissingIndicator`)
Agar tum numerical feature mein median impute kar rahe ho, lekin yeh bhi record karna chahte ho ki value missing thi:
```python
from sklearn.impute import MissingIndicator

# Parallel boolean indicator feature:
indicator = MissingIndicator()
missing_flags = indicator.fit_transform(X_train[continuous_features])
# Creates a boolean array: True if missing, False if observed
```

---

## 🔴 Section 4: The Cardinal Sin — Data Leakage During Imputation

> ⚠️ **THE GOLDEN SPLIT RULE (MANDATORY IN EVERY INTERVIEW & PROJECT):**
> Kabhi bhi train-test split karne se pehle poore dataset par `df.fillna()` mat lagao!
> Agar tumne poore dataset ka median nikala, toh **Test Set ke future instances ki information Train Set ke andar chali gayi**.
> Asli production mein jab kal naya customer aayega, toh tumhare model ke paas uska data pehle se nahi hoga.
> Model local machine par 98% accuracy dikhayega, par production deployment ke pehle din hi crash ho jayega!

```
❌ WRONG (DATA LEAKAGE):
Dataset -> Impute full dataset (Median/Mean) -> Train/Test Split

✅ CORRECT (ZERO LEAKAGE):
Dataset -> Train/Test Split (Stratified) -> Fit Imputer on TRAIN ONLY -> Transform TEST
```

---

## 📋 Section 5: Standard Step 03 Code Protocol (Procedural)

Is step par hum sirf **Target column ke missing values** ko drop karte hain aur dataset ka missing audit karte hain. Features ko impute karne ka kaam Step 07 (Train-Test Split) ke baad **Step 08 (Preprocessing)** mein kiya jata hai taaki Data Leakage na ho:

```python
# ==============================================================================
# STEP 3: MISSING VALUES AUDIT & TARGET CLEANING PROTOCOL
# ==============================================================================

# 1. Total Missing Values Audit
missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()

print("=" * 60)
print("🔍 STEP 3: MISSING VALUES AUDIT")
print("=" * 60)
print(f"Total Missing Values in Dataset: {total_missing:,}")

if total_missing > 0:
    cols_with_nulls = missing_counts[missing_counts > 0]
    print("\nColumns with Missing Values:")
    for col, cnt in cols_with_nulls.items():
        pct = (cnt / len(df)) * 100
        print(f"  • {col:<20}: {cnt} missing ({pct:.2f}%)")
else:
    print("✅ No missing values found in the dataset!")

# 2. Non-Negotiable: Drop Rows Where Target is Missing (If any)
target_col = 'pass'  # Set your target column name
if df[target_col].isnull().sum() > 0:
    initial_rows = len(df)
    df.dropna(subset=[target_col], inplace=True)
    dropped = initial_rows - len(df)
    print(f"\n⚠️ Dropped {dropped} unlabelled target rows. Clean dataset shape: {df.shape}")
else:
    print(f"✅ Target column '{target_col}' has zero missing values!")
print("=" * 60)
```

> 💡 **Agla Kadam:** Features ki missing value imputation (Median / Most Frequent) hum **Step 07 (Train-Test Split)** ke baad **Step 08 (Feature Preprocessing)** mein karenge, jisse test data ka median train mein leak na ho! (End-to-End Production Pipeline architecture **Guide 12** mein detailed cover ki gayi hai).

---

## 🧠 Universal Interview Flashcard: Missing Data Handling

| Question | Junior Response ❌ | Senior Developer Response ✅ |
| :--- | :--- | :--- |
| **"Missing values kaise handle karte ho?"** | `df.fillna(df.mean())` chala deta hoon | Pehle MCAR/MAR/MNAR nature aur missing percentage dekhta hoon. Target missing ho toh drop row, numerical mein median, categorical mein mode ya 'Missing' category. Aur leakage rokhne ke liye train-test split ke baad pipeline mein impute karta hoon. |
| **"Mean kab use karte ho aur Median kab?"** | Dono ek hi baat hain | Agar data symmetric bell-curve ho toh Mean, agar outliers ya skewed distribution ho toh Median (robust to outliers). |
| **"High missing (>60%) feature ko kya drop karna chahiye?"** | Haan, hamesha drop kar do | Hamesha nahi! Agar feature business-critical ya informative (MNAR) hai, toh `MissingIndicator` ya `'Missing'` category bana kar retain karte hain. |
| **"Train-test split se pehle imputation kyu nahi kar sakte?"** | Pata nahi, code chalta toh hai | Data Leakage hota hai! Test set ka mean/median train set ke statistical parameters ko pollute kar deta hai. |
