# 🎯 Classification Memory Notes: 01 - Libraries Import & Data Ingestion (Basic to Advanced)

> **The Senior Developer's First Rule:**
> Ek junior developer bina soche-samjhe kahin se bhi code copy-paste karke random libraries import kar leta hai (jaise tabular medical data mein text vectorizer `TfidfVectorizer` import kar lena!).
> Ek **Senior ML Engineer** pehle project ke data domain ko dekhta hai, structured tarike se imports ko categorize karta hai, aur data loading mein aane wale saare real-world traps (`?`, `-9`, encoding, multi-file sources) ko pehle hi line mein sambhal leta hai!

---

## 🧭 The Data Ingestion Decision Tree

```
                            RAW CLASSIFICATION DATA
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
TABULAR DATA (Numbers/Categories)    TEXT DATA (NLP)          MULTI-SOURCE DATA
(Jaise: Heart Disease, Churn, Credit) (Jaise: Spam, Reviews)   (Jaise: UCI Multi-Hospital)
- StandardScaler / SimpleImputer     - TfidfVectorizer        - combine / glob / concat
- OneHotEncoder                      - Regex / String utils   - na_values=['?', '-9']
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       DATA INGESTION SANITY CHECK
                       1. File exists? (`Path.exists()`)
                       2. Encoding trap? (`latin-1` / `utf-8`)
                       3. Hidden Nulls? (`?`, `-9` converted to NaN)
                       4. Shape & Head check in first 3 seconds!
```

---

## 🟢 Level 1: Senior Developer's Modular Library Imports

Imports ko hamesha **6 logical blocks** mein divide karo. Isse code padhne mein aasan, professional aur clean banta hai:

```python
# ==============================================================================
# 1. CORE DATA MANIPULATION & PATH UTILITIES
# ==============================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress unnecessary runtime warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 2. DATA VISUALIZATION & AESTHETICS
# ==============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 10

# ==============================================================================
# 3. FEATURE PREPROCESSING & DATA SPLITTING
# ==============================================================================
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ==============================================================================
# 4. CLASSIFICATION ALGORITHMS
# ==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# ==============================================================================
# 5. EVALUATION METRICS (THE CLASSIFICATION SUITE)
# ==============================================================================
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

# ==============================================================================
# 6. PIPELINE ORCHESTRATION & SERIALIZATION
# ==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

print("✅ Enterprise ML Libraries imported successfully.")
```

> [!TIP]
> **Domain-Specific Rule:**
> - Agar tabular numerical/categorical data hai (jaise Heart Disease ya Customer Churn), toh `StandardScaler` aur `SimpleImputer` zaroori hain, `TfidfVectorizer` nahi!
> - Agar NLP text data hai (jaise Spam Detection), tab `TfidfVectorizer` import karo.

---

## 🟡 Level 2: Basic to Advanced Data Loading (The 5 Real-World Traps)

### 1. Robust Path Handling (`pathlib.Path`)
Kabhi bhi raw strings jaise `"C:\\Users\\data\\file.csv"` mat likho (Linux aur Windows mein slash ka lafda ho jata hai). Hamesha `Pathlib` use karo:

```python
data_path = Path("data/heart+disease/heart.csv")

if not data_path.exists():
    raise FileNotFoundError(f"❌ File not found at: {data_path.resolve()}")

df = pd.read_csv(data_path)
print(f"Data Loaded! Shape: {df.shape}")
```

---

### 2. The Hidden Nulls Trap (`na_values`)
Medical aur Government datasets (jaise UCI Heart Disease) mein missing values khali nahi hoti! Doctors aksar missing test ko `'?'`, `'-9'`, ya `'Unknown'` likh dete hain!
Agar tumne sidha load kiya, toh Pandas inko **String (Text)** maan lega aur tumhare saare numbers kharab ho jayenge!

```python
# Solution: na_values parameter me saare ajeeb symbols de do
df = pd.read_csv(
    data_path,
    na_values=['?', '-9', '-9.0', 'Unknown', 'missing', 'NA', '']
)
# Ab Pandas in sabhi ko automatically असली NaN (Missing) me convert kar dega!
```

---

### 3. Special Characters & Encoding Trap (`encoding`)
Agar file load karte waqt error aaye: `UnicodeDecodeError: 'utf-8' codec can't decode...`
Toh file mein UK/European currency symbols ya foreign characters hain:

```python
# Solution: latin-1 ya ISO-8859-1 encoding pass karo
df = pd.read_csv(data_path, encoding='latin-1')
```

---

### 4. Custom Delimiters Trap (`sep`)
Aksar European financial ya medical data comma (`,`) ki jagah semicolon (`;`) ya Tab (`\t`) se alag hota hai:

```python
df = pd.read_csv(data_path, sep=';')   # Semicolon separated
df = pd.read_csv(data_path, sep='\t')  # Tab separated (.tsv)
```

---

### 5. Multi-Center Hospital / Multi-File Combining (Advanced Pattern)
UCI Heart Disease dataset jaise projects mein data 4 alag-alag hospitals se aata hai:
- Cleveland Clinic (`processed.cleveland.data`)
- Hungarian Institute of Cardiology (`processed.hungarian.data`)
- University Hospital Zurich, Switzerland (`processed.switzerland.data`)
- V.A. Medical Center, Long Beach, CA (`processed.va.data`)

Inhe combine karne ka professional automated pattern:

```python
data_dir = Path("data/heart+disease")
source_files = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data"
]

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

dataframes = []
for file in source_files:
    file_path = data_dir / file
    if file_path.exists():
        temp_df = pd.read_csv(file_path, names=columns, na_values=['?', '-9', '-9.0'])
        dataframes.append(temp_df)

# Saare hospital records ko ek master dataframe me jod do
df_combined = pd.concat(dataframes, ignore_index=True)
print(f"✅ Combined {len(dataframes)} hospital sources! Total rows: {len(df_combined):,}")
```

---

## 🔴 Level 3: Post-Load 3-Second Senior Health Audit

Data load hone ke theek baad, ek senior engineer hamesha yeh 5 checks run karta hai:

```python
def senior_data_audit(df, target_col='target'):
    print("=" * 60)
    print("📋 SENIOR DEVELOPER DATA INGESTION AUDIT")
    print("=" * 60)
    
    # 1. Shape check
    print(f"1. Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 2. Duplicate check
    dups = df.duplicated().sum()
    print(f"2. Duplicate Rows: {dups:,} ({dups/len(df)*100:.2f}%)")
    
    # 3. Missing Values Summary
    total_nulls = df.isnull().sum().sum()
    print(f"3. Total Missing Values: {total_nulls:,}")
    if total_nulls > 0:
        null_cols = df.isnull().sum()[df.isnull().sum() > 0]
        print(f"   Columns with Nulls:\n{null_cols}")
        
    # 4. Target Variable Check
    if target_col in df.columns:
        print(f"\n4. Target '{target_col}' Class Counts:")
        print(df[target_col].value_counts(dropna=False))
        print("\n   Class Proportions (%):")
        print((df[target_col].value_counts(normalize=True, dropna=False) * 100).round(2))
    else:
        print(f"⚠️ Target column '{target_col}' not found in columns!")
        
    # 5. Data Types breakdown
    num_cols = df.select_dtypes(include=['number']).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    print(f"\n5. Feature Types: {num_cols} Numerical, {cat_cols} Categorical")
    print("=" * 60)

# Run audit immediately after loading
senior_data_audit(df, target_col='target')
```

---

## 🧠 Quick Revision Checklist: Imports & Ingestion
- [ ] Imports ko 6 logical categories mein divide kiya?
- [ ] Sirf wahi libraries import ki jo is project ke data domain ko chahiye?
- [ ] Hardcoded string path ke bajaye `pathlib.Path` use kiya?
- [ ] Hidden missing symbols (`na_values=['?', '-9']`) handle kiye?
- [ ] CSV load hone ke turant baad 3-second Health Audit chalaya?

