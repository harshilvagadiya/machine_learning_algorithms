# 🎯 Classification Memory Notes: 02 - Data Types Audit & Feature Segregation

> **The Senior Developer's Golden Rule:**
> Regression mein data type galat hone par slope thoda shift ho sakta hai, lekin **Classification** mein data type ki galti model ke **Decision Boundary** aur **Weights** ko tabah kar deti hai!
> Machine learning model ke paas medical ya business domain ki samajh nahi hoti — woh sirf $z = w_1 x_1 + w_2 x_2 + \dots + b$ compute karta hai.
> Agar tumne Chest Pain Type (`cp: 1, 2, 3, 4`) ko continuous number samajh kar model mein bhej diya, toh model sochega ki *"Type 4 pain, Type 1 pain se 4 guna zyada hai!"* — jo ki clinically aur mathematically bilkul galat hai!

---

## 🧭 The Feature Classification Decision Tree

Asli classification tabular datasets mein features ko Pandas ke dtypes (`int64`, `float64`, `object`) ke bharose mat chhodo. Unhe **Domain Math** ke hisaab se classify karo:

```
                            RAW CLASSIFICATION FEATURES
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
TRUE CONTINUOUS (Scale)         DISCRETE CATEGORICAL (Code)     ID / LEAKAGE NOISE
(Age, BP, Cholesterol, Salary)  (Sex, Pain Type, Slope, State)  (Patient_ID, Row_Num)
- Infinite/Many possible values - Fixed distinct labels/enums   - Unique per row
- `nunique() > 10`              - `nunique() <= 10`             - `nunique() == len(df)`
         │                               │                               │
         ▼                               ▼                               ▼
    ACTION NEEDED                   ACTION NEEDED                   ACTION NEEDED
  StandardScaler /               One-Hot Encoding /              Drop immediately!
   RobustScaler                   Ordinal Encoding                (Overfitting risk)
```

---

## 🟢 Level 1: Basic Inspection (Kundali Dekhna)

Project shuru hote hi pehle 5 seconds mein data types ki basic health inspect karo:

### 1. Dtypes Summary & Count
```python
import pandas as pd
import numpy as np

# Total columns kitne numbers hain aur kitne objects/text hain?
print("--- Data Types Distribution ---")
print(df.dtypes.value_counts())
```

### 2. Full Overview (`df.info()`)
```python
# Non-null count + Memory usage + Data type
df.info()
```

### 3. `head()` vs `sample()` Trap
> **Senior Tip:** Sirf `df.head()` dekh kar khush mat ho! Pehli 5 rows aksar saaf hoti hain. Random rows dekhne ke liye hamesha `df.sample()` use karo:
```python
# Sample se beech ke ajeeb values (e.g. '?', 'fixed', -9) pakad mein aate hain
df.sample(5, random_state=42)
```

---

## 🟡 Level 2: The Senior Cardinality Filter (Continuous vs Categorical Separation)

Junior developer `df.select_dtypes(include=[np.number])` run karke saare numbers ko ek saath scale kar deta hai. Yeh tabular classification mein **sabse badi galti** hai!

### Kyu?
Clinical datasets (jaise Heart Disease) mein:
- `sex` (0, 1) $\to$ `float64`
- `cp` (1, 2, 3, 4) $\to$ `float64`
- `restecg` (0, 1, 2) $\to$ `float64`
- `slope` (1, 2, 3) $\to$ `float64`
- `ca` (0, 1, 2, 3) $\to$ `float64`

Yeh sab Pandas ke liye numbers hain, lekin asal mein yeh **Discrete Categories** hain!

### The Senior Automated Separation Pattern:
Cardinality (`nunique()`) ka use karke features ko automatically 3 buckets mein baanto:

```python
def categorize_features(df, target_col='target', cardinality_threshold=10):
    """
    Features ko True Continuous, Categorical, ID/Drop, aur Target mein separate karta hai.
    """
    continuous_features = []
    categorical_features = []
    drop_features = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        n_unique = df[col].nunique(dropna=True)
        col_type = df[col].dtype
        
        # 1. ID check (Har row mein alag value)
        if n_unique == len(df) and n_unique > 50:
            drop_features.append(col)
        # 2. String/Object types ya low cardinality numbers (<= 10 distinct values)
        elif col_type == 'object' or col_type.name == 'category' or n_unique <= cardinality_threshold:
            categorical_features.append(col)
        # 3. High cardinality numbers (Age, Chol, BP, Salary)
        else:
            continuous_features.append(col)
            
    print(f"✅ Continuous Features   ({len(continuous_features)}): {continuous_features}")
    print(f"✅ Categorical Features  ({len(categorical_features)}): {categorical_features}")
    if drop_features:
        print(f"⚠️ Potential ID Features ({len(drop_features)}): {drop_features}")
        
    return continuous_features, categorical_features, drop_features

# Run separation
continuous_features, categorical_features, drop_features = categorize_features(df, target_col='target')
```

---

## 🔴 Level 3: The 5 Deadly Data Type Traps in Tabular Classification

---

### ⚠️ Trap 1: The "Fake Number" (Coded Nominal Category) Trap

#### Problem:
Medical datasets mein categories ko numbers mein code kiya jata hai:
- `cp` (Chest Pain Type):
  - `1`: Typical Angina
  - `2`: Atypical Angina
  - `3`: Non-anginal pain
  - `4`: Asymptomatic

Agar tumne ise raw number ki tarah Logistic Regression mein daala:
$$z = w \cdot cp + \dots$$
Model assume karega ki:
$$\text{Type } 4 = 2 \times \text{Type } 2 = 4 \times \text{Type } 1$$
Jabki Type 4 (Silent/Asymptomatic) ka Type 1 se koi linear mathematical relation nahi hai! Isse model ke coefficients aur odds ratios completely distort ho jaate hain.

#### Solution:
Inhe categorical recognize karo aur **One-Hot Encoding** (`pd.get_dummies` ya Scikit-Learn `OneHotEncoder(drop='first')`) karo:
```python
# Before training, ensure it's treated as categorical
df['cp'] = df['cp'].astype('category')
```

---

### ⚠️ Trap 2: The "Float Binary Flag" Trap (Silent NaN Side-Effect)

#### Problem:
Tumne dekha ki `sex`, `fbs` (fasting blood sugar > 120), aur `exang` (exercise induced angina) 0 aur 1 hone ke bawajood `float64` kyun bane hue hain?
```python
print(df['sex'].dtype) # Output: float64
```
**Kyu hota hai?**
Standard NumPy aur Pandas mein agar kisi integer column mein **ek bhi NaN (missing value)** aa jaye, toh integer type NaN store nahi kar sakta! Pandas poore column ko chupke se `float64` (`0.0, 1.0, np.nan`) mein convert kar deta hai.

#### Solution:
Pehle Missing Values ko treat/impute karo, fir unhe cleanly integer ya boolean mein cast karo:
```python
# Imputation ke baad clean int banayein
df['sex'] = df['sex'].astype(int)
```
*(Modern Pandas alternative: `df['sex'] = df['sex'].astype('Int64')` — capital 'I' wala nullable integer bina imputation ke bhi NaN rakh sakta hai).*

---

### ⚠️ Trap 3: The "Mixed Data Types in a Single Column" Trap (The Multi-Hospital Disaster)

#### Problem:
Real-world datasets aksar multiple hospitals, branches ya vendors se combine kiye jaate hain.
Example: **UCI Heart Disease** dataset 4 alag-alag hospitals se aaya hai:
- Cleveland hospital ne `thal` ko numbers mein store kiya: `3.0, 6.0, 7.0`
- Long Beach / Hungarian hospital ne `thal` ko text mein store kiya: `'normal', 'fixed', 'reversible'`

Result:
```python
print(df['thal'].value_counts(dropna=False))
# Output:
# 3.0           166
# 7.0           117
# reversible     75
# normal         30
# fixed          28
# 6.0            18
# NaN           486
```
Ek hi column ke andar strings aur numbers mix ho gaye! Pandas ne ise `object` bana diya. Agar bina clean kiye Scikit-Learn mein daala toh model crash ho jayega.

#### Solution (Senior Mapping Dictionary):
Domain mapping dictionary bana kar dono formats ko ek standard label par align karo:
```python
# Clinical meaning:
# 3.0 / 3 -> 'normal'
# 6.0 / 6 -> 'fixed' (fixed defect)
# 7.0 / 7 -> 'reversible' (reversible defect)

thal_standardization = {
    '3.0': 'normal',
    3.0: 'normal',
    '3': 'normal',
    'normal': 'normal',
    
    '6.0': 'fixed',
    6.0: 'fixed',
    '6': 'fixed',
    'fixed': 'fixed',
    
    '7.0': 'reversible',
    7.0: 'reversible',
    '7': 'reversible',
    'reversible': 'reversible'
}

# Standardize column cleanly
df['thal'] = df['thal'].map(thal_standardization)

print("Standardized thal counts:")
print(df['thal'].value_counts(dropna=False))
```

---

### ⚠️ Trap 4: Hidden Text / Currency / Percentage Symbols in Numbers

#### Problem:
Financial ya Medical billing data mein numbers formatted text hote hain: `"$1,450.50"`, `"98.6%"`, ya whitespace `" 120 "`.
Pandas ise `object` samajhta hai.

#### Solution (The 2-Step Sanitizer):
```python
# 1. Regex se unwanted symbols hatao
clean_series = df['Salary'].astype(str).str.replace(r'[$,% ]', '', regex=True)

# 2. Safe numeric conversion (jo parse na ho sake, woh NaN ban jaye)
df['Salary'] = pd.to_numeric(clean_series, errors='coerce')
```

---

### ⚠️ Trap 5: The Target Variable Dtype Trap

#### Problem:
1. Target column `float64` nahi hona chahiye (e.g. `0.0, 1.0`). Scikit-Learn ke classification metrics aur ROC-AUC curves strict integer expect karte hain.
2. Clinical datasets mein raw target aksar **Multiclass** hota hai:
   - UCI Heart Disease mein raw `num` target tha: `0` (Healthy), `1` (Mild), `2` (Moderate), `3` (Severe), `4` (Critical).
   - Binary classification ke liye hume $0$ (Healthy) vs $\ge 1$ (Heart Disease present) chahiye.

#### Solution:
```python
# Binary conversion & strict int64 casting
df['target'] = (df['target'] > 0).astype(int)

# Sanity verification
print("Target dtype:", df['target'].dtype)
print("Target unique values:", df['target'].unique()) # Must be array([0, 1])
```

---

## 🟣 Level 4: Memory Optimization & Modern Pandas Types

Bade datasets (10 lakh+ rows) mein data types optimize karne se RAM consumption **70% se 85% tak drop** ho sakti hai:

```python
def optimize_dtypes(df):
    """
    Downcasts numeric types and converts low-cardinality strings to 'category'.
    """
    initial_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Float downcasting: float64 -> float32
        if col_type == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        # Integer downcasting: int64 -> int16 / int8
        elif col_type == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        # Object with low cardinality -> category
        elif col_type == 'object' and df[col].nunique() < 50:
            df[col] = df[col].astype('category')
            
    final_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory reduced: {initial_mem:.2f} MB -> {final_mem:.2f} MB ({100*(initial_mem-final_mem)/initial_mem:.1f}% saved)")
    return df
```

---

## 📋 The "Copy-Paste" Senior Developer Production Template

Kisi bhi classification project ke pehle cell mein yeh structured audit block run karo:

```python
# ==============================================================================
# STEP 2: DATA HYGIENE - D: DATA TYPES AUDIT & FEATURE SEGREGATION
# ==============================================================================

print("=" * 60)
print("🔍 SENIOR DATA TYPES AUDIT")
print("=" * 60)

# 1. Dtypes Breakdown
print("\n1. Data Types Distribution:")
print(df.dtypes.value_counts())

# 2. Automated Feature Segregation via Domain Cardinality
CARDINALITY_THRESHOLD = 10
target_col = 'target'

continuous_features = []
categorical_features = []

for col in df.columns:
    if col == target_col:
        continue
    n_unique = df[col].nunique(dropna=True)
    if df[col].dtype == 'object' or df[col].dtype.name == 'category' or n_unique <= CARDINALITY_THRESHOLD:
        categorical_features.append(col)
    else:
        continuous_features.append(col)

print(f"\n2. Feature Segregation:")
print(f"   • Continuous Features  ({len(continuous_features)} cols) -> StandardScaler required:")
print(f"     {continuous_features}")
print(f"   • Categorical Features ({len(categorical_features)} cols) -> One-Hot / Label Encoding required:")
print(f"     {categorical_features}")
print(f"   • Target Feature: '{target_col}' (Dtype: {df[target_col].dtype})")

# 3. Target Sanity Check
assert set(df[target_col].unique()).issubset({0, 1}), "❌ Target must only contain 0 and 1!"
assert np.issubdtype(df[target_col].dtype, np.integer), "❌ Target dtype must be integer!"
print("   • Target Status: ✅ Perfectly binary & integer!")
print("=" * 60)
```

---

## 🧠 Quick Revision Flashcard

| S.No | Scenario / Question | Junior Mistake ❌ | Senior Developer Solution ✅ |
| :---: | :--- | :--- | :--- |
| **1** | Chest Pain type (1, 2, 3, 4) in numeric col | Seedha `StandardScaler` laga kar model mein pass kar diya | Coded category hai! One-Hot Encoding (`pd.get_dummies`) karo |
| **2** | Binary flags (`sex`, `fbs`) are `float64` | Socha ki decimal number hai | Samjho ki missing value ki wajah se float bana hai, impute karke `int` karo |
| **3** | Single column mein numbers (`3.0`) aur strings (`'normal'`) hain | Error aane par crash ho gaya ya rows drop kar di | Domain dictionary se sabko standard string labels mein `.map()` karo |
| **4** | Feature separation | Sirf `select_dtypes(include=number)` kiya | `nunique() <= 10` cardinality check karke discrete codes ko pakda |
| **5** | Target column | `float64` ya string rehne diya | Strictly `int64` binary `(0, 1)` banaya |

