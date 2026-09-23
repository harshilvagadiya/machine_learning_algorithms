# 📖 ML Memory Notes: M - Missing Values (Basic to Advanced)

> **Golden Law of Missing Values:** 
> Agar **Target ($y$)** missing hai, toh **AANKH BAND KARKE DROP KARO**! 
> Agar **Features ($X$)** missing hain, toh **Percentage aur Business Context** dekh kar faisla karo (Drop vs. Impute)!

---

## 🧭 The Master Decision Tree (Visual Guide)

```
                       MISSING VALUE MILI?
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
   TARGET (y) MEIN HAI?                            FEATURE (X) MEIN HAI?
        │                                               │
        ▼                                               ▼
  🔥 ALWAYS DROP ROW!                             KITNA PERCENT MISSING HAI?
  `df.dropna(subset=['target'])`                        │
                                        ┌───────────────┴───────────────┐
                                        ▼                               ▼
                                  > 60% - 70% GAYAB                 < 10% - 20% GAYAB
                                        │                               │
                                        ▼                               ▼
                                  DROP COLUMN!                     IMPUTE (BHARO)!
                                  `df.drop(columns=['col'])`            │
                                                        ┌───────────────┴───────────────┐
                                                        ▼                               ▼
                                                   NUMERIC DATA                    TEXT / CATEGORY
                                                        │                               │
                                            ┌───────────┴───────────┐                   ▼
                                            ▼                       ▼             MODE (Most Frequent)
                                         OUTLIERS HAIN?         NO OUTLIERS       `fillna(df[col].mode()[0])`
                                            │                       │
                                            ▼                       ▼
                                         MEDIAN (Middle)         MEAN (Average)
                                         `fillna(median)`        `fillna(mean)`
```

---

## 🟢 Level 1: Detection & Counting (Khali Dabbe Dhoondna)

### 1. The Standard 3-Line Detection Formula
```python
import pandas as pd

# 1. Total nulls count karo
null_series = df.isnull().sum()

# 2. Sirf unko rakho jo 0 se bade hain (> 0) aur descending order mein lagao
missing_only = null_series[null_series > 0].sort_values(ascending=False)

# 3. Percentage nikalo
missing_pct = (missing_only / len(df)) * 100

# 4. Beautiful summary table
missing_table = pd.DataFrame({
    "Missing Count": missing_only,
    "Percentage (%)": missing_pct.round(2)
})

print("--- Missing Values Report ---")
if len(missing_table) > 0:
    display(missing_table)
else:
    print("🎉 Zero missing values! Data 100% complete hai.")
```

---

## 🟡 Level 2: Visualizing Missing Values (Graphs)

Data ko aankhon se dekhna sabse aasan hota hai ki khali dabbe kahan hain:

### Visual 1: Missing Percentage Bar Chart
```python
import matplotlib.pyplot as plt

if len(missing_table) > 0:
    plt.figure(figsize=(9, 4))
    bars = plt.barh(missing_table.index[::-1], missing_table["Percentage (%)"][::-1], color="#d62728", edgecolor="black")
    plt.title("Missing Data Percentage by Feature", fontsize=12, fontweight="bold")
    plt.xlabel("Percentage (%) Missing")
    plt.xlim(0, 100)
    
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontweight="bold")
    plt.tight_layout()
    plt.show()
```

### Visual 2: Missing Data Heatmap (Bina kisi extra library ke!)
```python
# Pure dataset mein kahan-kahan holes hain, yellow line ban ke dikhega!
plt.figure(figsize=(10, 4))
plt.imshow(df.isnull(), cmap="viridis", aspect="auto")
plt.title("Missing Values Heatmap (Yellow = Missing, Purple = Present)", fontsize=11, fontweight="bold")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.colorbar(label="Is Null (1=True, 0=False)")
plt.tight_layout()
plt.show()
```

---

## 🔴 Level 3: Imputation Strategies (Kaun Sa Formula Kab?)

### 1. MEAN vs. MEDIAN vs. MODE (The Golden Rule)

| Type | When to Use? | Code Example | Why? (Desi Logic) |
| :--- | :--- | :--- | :--- |
| **Median** | Numeric data jisme **Outliers** ya Skewness ho (Income, Price, Area) | `df['Salary'] = df['Salary'].fillna(df['Salary'].median())` | Ambani class mein aa jaye toh average bigad jata hai, median same rehta hai! |
| **Mean** | Numeric data jo **Symmetric / Normal** ho (Height, Body Temp) | `df['Age'] = df['Age'].fillna(df['Age'].mean())` | True mathematical balance point. |
| **Mode** | Categorical / Text data ('Male'/'Female', 'City') | `df['City'] = df['City'].fillna(df['City'].mode()[0])` | Jo sabse zyada baar repeat hua, wahi sabse likely answer hai. |
| **Constant (0)** | Jab missing hone ka matlab **Absence** ho (No Garage, No Pool) | `df['GarageCars'] = df['GarageCars'].fillna(0)` | Agar ghar mein garage nahi hai, toh capacity = 0 cars. |

---

### 2. Pro Level: Scikit-Learn Pipeline Imputation (Data Leakage Se Bacho!)

> ⚠️ **Data Leakage Warning:** 
> Test data par kabhi bhi `.fit()` nahi karte! Padhai train se karo, test par wahi number chipkao!

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1. Pehle Split karo!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Imputer banao (Median for numbers)
imputer = SimpleImputer(strategy="median")

# 3. Train par: Seekho + Badlo (fit_transform)
X_train_clean = imputer.fit_transform(X_train)

# 4. Test par: Sirf Badlo (transform)
X_test_clean = imputer.transform(X_test)
```

---

## 🟣 Level 4: The "Hidden Nulls" Trap (Chhupe Hue Bhoot!)

Asli company ke data mein log khali jagah ko `NaN` nahi chhodte, ajeeb cheezein likh dete hain:
- `"?"`
- `"None"` / `"NULL"` / `"N/A"`
- `"-999"` / `"-1"` (Database dummy numbers)
- `"   "` (Blank spaces)

Python sochta hai yeh normal text ya number hai!

### 💡 Unmasking Weapon (Replace With Real NaN):
```python
import numpy as np

# Sabhi chhupe hue dabbo ko asli NaN bana do!
df = df.replace(["?", "None", "N/A", "NA", -999, -1, "   ", ""], np.nan)

# Ab dobara check karo:
print(df.isnull().sum())
```

---

## 📋 The "Copy-Paste" Template for Any Project

Har project ke EDA mein yeh cell copy kar lo:

```python
# --- M: MISSING VALUES AUDIT & SUMMARY ---

null_counts = df.isnull().sum()
missing_only = null_counts[null_counts > 0].sort_values(ascending=False)

if len(missing_only) > 0:
    missing_pct = (missing_only / len(df)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing_only,
        "Percentage (%)": missing_pct.round(2)
    })
    print(f"⚠️ Alert: {len(missing_summary)} columns contain missing values:")
    display(missing_summary)
else:
    print("✅ Perfect! Zero missing values detected across all columns.")
```
