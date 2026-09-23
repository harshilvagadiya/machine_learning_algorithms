# 🎯 Classification Memory Notes: 04 - Duplicate Records Audit & Purging

> **The Senior Developer's Golden Rule of Duplicates:**
> Agar dataset mein duplicate rows hain aur tumne train-test split kar diya, toh ek duplicate row Train set mein chali jayegi aur doosri Test set mein!
> Model us patient/record ko Train mein **memorize (yaad)** kar lega aur Test mein 100% accuracy de dega.
> Yeh **Data Leakage** hai! Asli production data aane par model fail ho jayega. Isliye duplicate rows ko train-test split se pehle hi hatana mandatory hai!
> **The Senior Developer's Universal Rule for Duplicate Records:**
> Yeh Memory Note kisi ek project ke liye nahi hai — yeh har classification project (Fraud, Churn, Medical, Sales) ke liye **Universal Standard Operating Procedure (SOP)** hai.
> 
> **The Golden Law:**
> **The Senior Developer's Golden Rule for Duplicate Records:**
> Duplicate records Machine Learning model ke liye **Zeher (Poison)** hote hain!
> Agar dataset mein duplicate rows hain aur tumne train-test split kar diya, toh ek duplicate copy Train set mein chali jayegi aur doosri Test set mein!
> Model us record ko Train mein **memorize (yaad)** kar lega aur Test evaluation mein fake 99% accuracy dikhayega.
> Yeh **Data Leakage** hai! Asli production data aane par model fail ho jayega. Isliye duplicate rows ko train-test split se pehle hatana **non-negotiable mandatory step** hai!
> Model us record ko Train mein **memorize (ratta)** kar lega aur Test evaluation mein fake 99% accuracy dikhayega (Data Leakage).
> Asli production data aane par model buri tarah crash ho jayega. Isliye duplicate rows ko train-test split se pehle dhoondna aur handle karna **non-negotiable step** hai!

---

## 🧭 The Duplicate Records Decision Tree
## 🧭 The Universal Duplicate Records Decision Tree
## 🧭 The Universal Duplicate Records Decision Tree (For Any Dataset)

```
                       DUPLICATE RECORDS AUDIT (`df.duplicated()`)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
          EXACT DUPLICATES (All Cols Match)               PARTIAL DUPLICATES (Subset Match)
          (e.g., Har feature same to same hai)            (e.g., Same Patient ID, different dates)
          (Har feature same-to-same hai)                  (e.g., Same Customer ID, different timestamps)
                    │                                               │
                    ▼                                               ▼
         🔥 DROP ROW IMMEDIATELY!                         DOMAIN INVESTIGATION:
         `df.drop_duplicates(keep='first')`               - Keep latest record?
                                                          - Feature engineer time delta?
         `df.drop_duplicates(keep='first')`               - Keep latest record? (`sort_values().drop_duplicates(keep='last')`)
                                                          - Groupby aggregate? (Total spend, average visits)
                            DUPLICATE AUDIT (`df.duplicated()`)
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         EXACT FULL-ROW DUPLICATES                       PARTIAL / SUBSET DUPLICATES
         (Har ek column 100% match karta hai)           (Same Entity/ID, different dates/features)
                     │                                               │
                     ▼                                               ▼
       TARGET (y) SAME HAI YA DIFFERENT?                 DATA TYPE & DOMAIN CONTEXT:
                     │                                               │
        ┌────────────┴────────────┐                      ┌───────────┴───────────┐
        ▼                         ▼                      ▼                       ▼
    SAME TARGET               DIFFERENT TARGET       TIME-SERIES / EVENTS    AGGREGATABLE
 (Identical Twins)          (Contradictory/Noise)    (Repeated Logins)       (Multiple Transactions)
        │                         │                      │                       │
        ▼                         ▼                      ▼                       ▼
   DROP COPY!               DROP BOTH COPIES!       KEEP LATEST!            GROUPBY & AGGREGATE!
`drop_duplicates(          `drop_duplicates(       `sort_values().drop_    `groupby('id').agg()`
  keep='first')`             keep=False)`           duplicates(keep='last')`
                       DUPLICATE AUDIT (`df.duplicated()`)
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
 EXACT FULL-ROW DUPLICATES                           PARTIAL / SUBSET DUPLICATES
 (Har ek column 100% match karta hai)               (Same Entity/ID, different dates/features)
             │                                                       │
             ▼                                                       ▼
 TARGET (y) SAME YA DIFFERENT?                       DATA TYPE & DOMAIN CONTEXT:
             │                                                       │
  ┌──────────┴──────────┐                            ┌──────────────┴──────────────┐
  ▼                     ▼                            ▼                             ▼
SAME TARGET         DIFFERENT TARGET            TIME-SERIES / EVENTS         AGGREGATABLE
(Identical Twins)   (Contradictory/Noise)       (Repeated Logins)            (Multiple Transactions)
  │                     │                            │                             │
  ▼                     ▼                            ▼                             ▼
DROP COPY!          DROP BOTH COPIES!           KEEP LATEST!                 GROUPBY & AGGREGATE!
`drop_duplicates(   `drop_duplicates(           `sort_values().drop_         `groupby('id').agg()`
  keep='first')`      keep=False)`               duplicates(keep='last')`
```

---

## 🟢 Level 1: Detection & Inspection
## 🟢 Section 1: Universal Plug & Play Duplicate Audit Function
## 🔬 Section 1: The 4 Real-World Types of Duplicates (Interviews & Architecture)
## 🔬 Section 1: The 4 Real-World Types of Duplicates

### 1. Count Duplicate Rows
Is function ko kisi bhi future project mein paste karo — yeh exact duplicate count, percentage, aur sample duplicate pairs display karta hai:
Asli production datasets mein duplicates sirf ek tarah ke nahi aate. Yahan 4 main scenarios aate hain:

### 1. Type 1: Exact Full-Row Duplicates (The 100% Clones)
* **Kyu aate hain?** Multiple CSV files ko galat concatenate kar diya, API request retry ho gayi, ya database ETL pipeline ne batch do baar run kar diya.
* **Impact:** Har feature ($X$) aur target ($y$) exact match karta hai.
* **Action:** Pehli occurrence ko rakho aur baaki saari duplicate copies drop karo (`keep='first'`).

### 2. Type 2: Entity / ID-Level Duplicates (Repeated Customer/Patient Events)
* **Kyu aate hain?** Ek hi customer ne 3 alag-alag mahino mein website visit ki ya ek hi patient 2 alag-alag din hospital aaya.
* **Impact:** Customer ID same hai, lekin timestamp, order amount, ya vitals alag hain.
* **Action:** 
* **Action:**
  * Agar classification current status par hai $\implies$ Sort karke **Latest Record** rakho (`keep='last'`).
  * Ya fir **Feature Engineering** karo: Groupby karke features banao (e.g. `total_visits`, `average_order_value`).

### 3. Type 3: The "Contradictory / Toxic" Duplicates (Same Features, Different Target!)
* **Sabse Khatarnak Scenario:** Row 10 aur Row 50 ke saare features $X$ identical hain, lekin:
* **Sabse Khatarnak Scenario:** Do rows ke saare features $X$ identical hain, lekin:
  * Row 10: `target = 0`
  * Row 50: `target = 1`
* **Kyu aate hain?** Human annotator disagreement, labeling error, ya missing critical hidden feature (e.g. genetic factor jo dataset mein recorded hi nahi tha).
* **Kyu aate hain?** Human annotator disagreement, labeling error, ya missing critical hidden feature.
* **Impact:** Model confuse ho jaata hai! Same inputs par do alag labels hone se loss function explode hota hai.
* **Action:** **DROP BOTH ROWS!** Aise ambiguous data ko dataset se nikaalna sabse safe hota hai (`keep=False`).
* **Action:** **DROP BOTH ROWS!** (`keep=False`) — Aise ambiguous data ko dataset se nikaalna sabse safe hota hai.

### 4. Type 4: Coincidental Duplicates (Low Dimensionality Trap)
* **Kyu aate hain?** Agar dataset mein sirf 3 ya 4 categorical features hon (e.g. `Gender: Male`, `City: Delhi`, `Education: Graduate`), toh 2 alag insaanon ke traits naturally identical ho sakte hain!
* **Kyu aate hain?** Agar dataset mein sirf 3–4 categorical features hon (e.g. `Gender: Male`, `City: Delhi`, `Education: Graduate`), toh 2 alag insaanon ke traits naturally identical ho sakte hain!
* **Action:** Agar features bahut kam hain, toh domain check karo ki kya yeh natural coincidence hai ya system error.

---

## 🟢 Section 2: Universal Plug & Play Duplicate Audit Function

Is function ko kisi bhi project ke Step 4 mein daalo. Yeh total duplicates, percentage, aur original vs duplicate pairs ko ek saath display karta hai:
Is function ko kisi bhi project ke Step 04 mein daalo. Yeh total duplicates, percentage, contradictory labels, aur ID duplicates ek saath display karta hai:

```python
import pandas as pd
import numpy as np

total_dups = df.duplicated().sum()
dup_pct = (total_dups / len(df)) * 100
print(f"Total Duplicate Rows: {total_dups:,} ({dup_pct:.2f}%)")
def universal_duplicate_audit(df, subset_cols=None):
def universal_duplicate_audit(df, target_col=None, id_cols=None):
    """
    Universally audits duplicate records for ANY dataset.
    Works for full-row duplicates or specific subset of columns (like Customer_ID).
    Universal Duplicate Audit for ANY tabular dataset.
    Audits full-row duplicates, ID duplicates, and contradictory target labels.
    """
    total_rows = len(df)
    dups_mask = df.duplicated(subset=subset_cols, keep='first')
    total_dups = dups_mask.sum()
    dup_pct = (total_dups / total_rows) * 100
    full_dups = df.duplicated().sum()
    dup_pct = (full_dups / total_rows) * 100
    

    print("=" * 70)
    print("📋 UNIVERSAL DUPLICATE RECORDS AUDIT REPORT")
    print("=" * 70)
    print(f"Total Rows:            {total_rows:,}")
    print(f"Duplicate Rows Count:  {total_dups:,} ({dup_pct:.2f}%)")
    print(f"Unique Rows Count:     {total_rows - total_dups:,}")
    print(f"Total Rows:                {total_rows:,}")
    print(f"Exact Full-Row Duplicates: {full_dups:,} ({dup_pct:.2f}%)")
    print(f"Unique Records:            {total_rows - full_dups:,}")
    
    if total_dups > 0:
        print("\n⚠️ Duplicate Records Detected! Displaying sample duplicate pairs:")
        # Show both the original and duplicate copy together
        dup_pairs = df[df.duplicated(subset=subset_cols, keep=False)].sort_values(
            by=list(subset_cols or df.columns[:2])
        )
        print(dup_pairs.head(6))
        print("\n⚡ Recommended Action: Run df.drop_duplicates(keep='first', inplace=True)")

    # 1. Inspect Exact Duplicate Pairs
    if full_dups > 0:
        print("\n🔍 Sample Duplicate Copies Found (Showing Original + Copy):")
        sample_dups = df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:2]))
        print(sample_dups.head(6))
    else:
        print("✅ 100% Clean! Dataset has ZERO duplicate records.")
        print("✅ Zero full-row duplicates found.")
        

    # 2. Check for Contradictory Labels (Same X, Different y)
    if target_col and target_col in df.columns:
        feature_cols = [c for c in df.columns if c != target_col]
        contradictory = df[df.duplicated(subset=feature_cols, keep=False)]
        
        # Check if identical features have differing target labels
        if len(contradictory) > 0:
            diff_targets = contradictory.groupby(feature_cols)[target_col].nunique()
            toxic_cases = diff_targets[diff_targets > 1]
            if len(toxic_cases) > 0:
                print(f"\n⚠️ CRITICAL ALERT: Found {len(toxic_cases)} groups of CONTRADICTORY labels!")
                print("   (Same features give both Class 0 and Class 1 -> Label Noise!)")
            else:
                print("✅ No contradictory target labels found.")
                

    # 3. Check for ID Duplicates (if ID column provided)
    if id_cols:
        id_dups = df.duplicated(subset=id_cols).sum()
        if id_dups > 0:
            print(f"\n⚠️ Entity Duplicates on {id_cols}: {id_dups:,} duplicate IDs found!")

    print("=" * 70)
    
    return total_dups
    return full_dups

# Usage for ANY future dataset:
# dups_count = universal_duplicate_audit(df)
# How to run for ANY future dataset:
# dups_count = universal_duplicate_audit(df, target_col='target')
```

### 2. View All Copies of Duplicate Rows
> **Senior Tip:** Sirf `df.duplicated()` mat chalao (woh sirf duplicate copy dikhata hai). Hamesha `keep=False` use karo taaki original aur duplicate dono ek saath compare ho sakein:
---

## 🟡 Section 3: The 3 Pandas Duplicate Functions You Must Master
## 🟡 Section 3: The 3 Pandas Duplicate Parameters You Must Master

Junior developers sirf `df.duplicated()` chala kar confuse ho jaate hain. Yahan Pandas ke 3 parameters ka deep intuition hai:
Junior developers sirf `df.duplicated()` chala kar confuse ho jaate hain. Yahan Pandas ke 3 `keep=` parameters ka deep intuition hai:

### 1. `keep='first'` (Default)
### 1. `keep='first'` (Default — Standard Deduplication)
Pehle aane wale record ko "Original" maanta hai aur baad wale saare identical records ko `True` (Duplicate) mark karta hai.
```python
if total_dups > 0:
    print("Inspecting Duplicate Pairs:")
    display(df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:3])))
# Baad wale saare duplicate clones drop karo:
df.drop_duplicates(keep='first', inplace=True)
```
## 🟡 Section 2: Universal Deduplication Scenarios & Actions

---
### 2. `keep='last'`
### 2. `keep='last'` (Time-Series / Latest Record)
Aakhri aane wale record ko "Original" maanta hai aur pehle walon ko mark karta hai.
* **Best for:** Time-series / Chronological data jahan latest status chahiye.
```python
# Latest customer status retain karne ke liye:
df.sort_values(by='timestamp', inplace=True)
df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)
```

## 🟡 Level 2: The Senior Action Plan (Drop Before Split)
### 📌 Scenario 1: Exact 100% Full-Row Duplicates
* **Cause:** Multiple file downloads, repeated API batch pulls, or database logging retries.
* **Code:**
  ```python
  # Safe drop keeping the first entry
  initial_count = len(df)
  df.drop_duplicates(keep='first', inplace=True)
  print(f"Cleaned {initial_count - len(df)} duplicates. New row count: {len(df):,}")
  ```

Hamesha pehle deduplicate karo:
### 3. `keep=False` (The Secret Diagnostic Weapon)
Yeh duplicate pair ke **dono copies (original aur duplicate dono)** ko `True` mark kar deta hai!
* **Best for:** Duplicate pairs ko aamne-saamne compare karne ke liye.
* **Best for:** Duplicate pairs ko aamne-saamne compare karne ke liye, ya contradictory rows ko drop karne ke liye.
```python
# Drop duplicates keeping the first occurrence
initial_len = len(df)
df.drop_duplicates(keep='first', inplace=True)
print(f"Removed {initial_len - len(df)} duplicate records. Clean row count: {len(df):,}")
# Dono copies ko ek saath screen par dekhna:
df[df.duplicated(keep=False)].sort_values(by='some_col')
df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:3]))

# Contradictory labels wali dono rows drop karne ke liye:
feature_cols = [c for c in df.columns if c != target_col]
df.drop_duplicates(subset=feature_cols, keep=False, inplace=True)
```

---

### 📌 Scenario 2: Partial Duplicates (Same Entity, Different Timestamps)
* **Example:** Customer Churn dataset jisme ek hi customer ID do baar aa rahi hai (ek January mein, ek March mein).
* **Code:**
  ```python
  # Sort by Date/Timestamp and keep the latest interaction
  df.sort_values(by=['customer_id', 'timestamp'], ascending=[True, True], inplace=True)
  df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)
  ```
## 🔴 Section 4: The Fatal Trap — Data Leakage via Duplicates

---
> ⚠️ **THE ARTIFICIAL 99% ACCURACY TRAP:**
> Agar dataset mein duplicate rows hain aur tumne train-test split chala diya:
> Agar dataset mein duplicate rows hain aur tumne train-test split pehle chala diya:
> ```python
> # ❌ DEADLY LEAKAGE ORDER!
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
> # Fir baad mein duplicates check kiye...
> # Fir baad mein duplicates check kiye... (Too Late!)
> ```
> **Kyu yeh gunah hai?**
> Man lo Patient #42 ki do identical copies thi (Row 15 aur Row 80).
> Random split mein Row 15 chali gayi **Train Set** mein aur Row 80 chali gayi **Test Set** mein!
> Model ne Patient #42 ke saare features aur target train set mein **memorize (ratta)** kar liye.
> Jab Test set mein wahi patient aaya, toh model ne 100% correct prediction de di!
> Tum sochoge mera model genius hai (99% accuracy!), lekin jab hospital mein naya patient aayega, toh accuracy 50% par gir jayegi!
> Tum sochoge mera model genius hai (99% accuracy!), lekin jab hospital mein naya patient aayega toh accuracy 50% par gir jayegi!

## 🔴 Level 3: Real-World Case Study (UCI Heart Disease)
### 📌 Scenario 3: Conflicting Target Labels for Identical Features
* **The Dangerous Edge Case:** Do rows ke saare features identical hain, lekin ek mein `target = 0` aur doosre mein `target = 1`!
* **Why it happens:** Human labeler disagreement or noise.
* **Code:**
  ```python
  # Check if identical feature rows have different target values
  feature_cols = [c for c in df.columns if c != target_col]
  conflicting = df[df.duplicated(subset=feature_cols, keep=False)]
  if len(conflicting) > 0:
      print(f"⚠️ Warning: Found {len(conflicting)} conflicting rows with contradictory labels!")
      # Safest industry solution: Drop ambiguous conflicting rows
      df.drop_duplicates(subset=feature_cols, keep=False, inplace=True)
  ```
### ✅ The Senior Order of Operations:
### ✅ The Mandatory Order of Operations:
```
STEP 1: LOAD DATA
STEP 3: Missing Values Audit & Target Cleaning
           │
           ▼
STEP 2: D - DATA TYPES AUDIT & SANITIZATION
           │
           ▼
STEP 3: D - DUPLICATE RECORDS REMOVAL (PRE-SPLIT!)  <-- Yahan hatana anivarya hai!
STEP 4: DUPLICATE RECORDS REMOVAL  <-- Yahan hatana anivarya hai! (PRE-SPLIT)
        `df.drop_duplicates(keep='first', inplace=True)`
           │
           ▼
STEP 4: TRAIN / TEST SPLIT (Zero Leakage Guarantee)
STEP 5: Target Analysis & Baseline
           │
           ▼
STEP 5: M - IMPUTATION & SCALING (Inside Cross-Validation / Pipeline)
STEP 7: TRAIN / TEST SPLIT (Zero Leakage Guarantee)
```

Humare Heart Disease dataset (920 rows) mein **exact 2 duplicate rows** hain:
- Ek Hungarian center se duplicate record aaya tha.
- Dono records mein har clinical attribute (Age, Sex, Cholesterol, BP, ECG, Target) identical tha.
- Agar hum ise nahi hatate, toh evaluation unbiased nahi rehti.
---

## 📋 Section 3: Universal Production-Ready Deduplication Template
## 📋 Section 5: Universal "Copy-Paste" Deduplication Template (For ANY Project)
## 📋 Section 5: Standard Step 04 Code Protocol (Procedural)

Har classification project ke Step 4 mein yeh robust block paste karo:
Har classification project ke Step 04 mein yeh clean block paste karo:

```python
# Verification in notebook:
# ==============================================================================
# UNIVERSAL STEP: D - DUPLICATE RECORDS ELIMINATION (FOR ANY PROJECT)
# UNIVERSAL STEP: D - DUPLICATE RECORDS AUDIT & ELIMINATION
# STEP 4: DATA HYGIENE - D: DUPLICATE RECORDS AUDIT & PURGING
# ==============================================================================

# 1. Audit duplicates
initial_rows = len(df)
dups = df.duplicated().sum()
assert dups == 0, f"❌ Dataset still contains {dups} duplicate records!"
print("✅ Dataset is 100% free of duplicate leakage.")
print("=" * 65)
print("🧹 SENIOR DEVELOPER DEDUPLICATION PROTOCOL")
print("=" * 65)

initial_rows = len(df)
dups = df.duplicated().sum()

if dups > 0:
    print(f"⚠️ Removing {dups:,} duplicate rows ({dups / initial_rows * 100:.2f}% of dataset)...")
# 1. Count duplicates before removal
initial_row_count = len(df)
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    pct = (duplicate_count / initial_row_count) * 100
    print(f"⚠️ Found {duplicate_count:,} duplicate records ({pct:.2f}% of dataset).")
    
    # 2. Inspect sample duplicate pairs
    print("\nSample duplicate rows being purged:")
    display(df[df.duplicated(keep=False)].head(4))
    
    # 3. Purge duplicates keeping the first instance
    print(f"⚠️ Found {dups:,} duplicate records ({dups / initial_rows * 100:.2f}% of dataset).")
    df.drop_duplicates(keep='first', inplace=True)
    
# 2. Production Assertion Check (Must be 0 before split)
assert df.duplicated().sum() == 0, "❌ CRITICAL: Dataset still contains duplicates!"
print(f"✅ Deduplication verified! Clean dataset ready for modeling: {len(df):,} rows.")
    cleaned_row_count = len(df)
    print(f"\n✅ Successfully removed {duplicate_count:,} duplicate records.")
    print(f"   Clean dataset shape: {cleaned_row_count:,} rows × {df.shape[1]} columns.")
    print(f"✅ Successfully removed {dups:,} duplicate records.")
    print(f"   Clean dataset shape: {len(df):,} rows × {df.shape[1]} columns.")
else:
    print("✅ Zero duplicate records found. Dataset is already clean!")
    print("✅ Zero duplicate records found. Dataset is already 100% free of duplicate leakage!")

# 4. Production Sanity Assertion
# Production Sanity Assertion (Must pass before proceeding to split)
assert df.duplicated().sum() == 0, "❌ CRITICAL: Duplicates still remain in dataset!"
print("=" * 65)
```

---

## 🧠 Quick Universal Interview Flashcard: Duplicates

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Duplicate rows ko kab drop karte hain?"** | Split ke baad ya kabhi bhi | **Train-Test split se pehle!** Warna duplicate rows train aur test mein divide hokar Data Leakage aur fake high test score dengi. |
| **"`keep='first'` aur `keep=False` mein kya farq hai?"** | Dono drop karte hain | `keep='first'` pehli copy chhod kar baaki copies mark karta hai; `keep=False` original aur duplicate dono copies ko mark karta hai (investigation ke liye best). |
| **"Agar do rows ke features same hon par target alag ho toh?"** | Kisi ek ko drop kar do | Yeh "Contradictory Label Noise" hai. Sabse safe tarika dono rows ko drop karna hai (`keep=False`) taaki model ambiguous data par confuse na ho. |
| **"Agar do rows ke features same hon par target alag ho toh?"** | Kisi ek ko drop kar do | Yeh "Contradictory Label Noise" hai. Sabse safe tarika dono rows drop karna hai (`keep=False`) taaki model ambiguous data par confuse na ho. |
| **"Partial duplicates kaise handle karte hain?"** | `drop_duplicates()` chala dete hain | Timestamp dekh kar latest record rakhte hain (`keep='last'`) ya feature engineering karke aggregate karte hain. |