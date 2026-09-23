# 📖 ML Memory Notes: Categorical Encoding & Feature Engineering (Basic to Advanced)

> **Golden Law of Features:** 
> **"Garbage In = Garbage Out!"** 
> ML model kitna bhi bada ho (Linear Regression ho ya Deep Learning), agar usko galat number ya kachra feature khilaoge, toh woh andaza bhi kachra hi lagayega!
> 
> Computer ko sirf **Numbers** samajh aate hain, English text nahi! Lekin text ko number mein badalne ke strict kanoon hote hain.

---

## 🧭 The Categorical Encoding Decision Tree

```
                    TEXT / CATEGORICAL COLUMN AAYA?
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
   KISI MEIN ORDER / RANK HAI?                     KOI ORDER NAHI HAI?
   (Low < Medium < High,                           (Gender: Male/Female,
   Matric < Bachelor's < Master's < PhD)           City: Delhi/Mumbai/Pune,
          │                                        Color: Red/Blue/Green)
          ▼                                               │
   ⭐ ORDINAL ENCODING                                    ▼
   (Manual Mapping via `.map()`)             KITNI UNIQUE CATEGORIES HAIN?
   `{"Low": 0, "Medium": 1, "High": 2}`                   │
                                          ┌───────────────┴───────────────┐
                                          ▼                               ▼
                                   CHHOTA CARDINALITY              HIGH CARDINALITY
                                      (<= 10 categories)             (> 15-20 categories)
                                          │                               │
                                          ▼                               ▼
                                  ⭐ ONE-HOT ENCODING              ⭐ TOP-N + "OTHER"
                                  `pd.get_dummies(...,            ya FREQUENCY ENCODING
                                   drop_first=True)`
```

---

## 🟢 Level 1: Ordinal Encoding (Jahan Rank / Hierarchy Hoti Hai)

Agar categories ke beech ek natural sequence ya ranking hai:
- Education: `High School` < `Bachelor's` < `Master's` < `PhD`
- Customer Rating: `Poor` (1) < `Average` (2) < `Good` (3) < `Excellent` (4)
- House Quality: `Ex` (5) > `Gd` (4) > `TA` (3) > `Fa` (2) > `Po` (1)

### Pro Method: Explicit Mapping (Sabse Safe & Transparent)
```python
# Hamesha dictionary banake map karo taaki tumhare control mein rahe ki kisko kya number mila:
quality_order = {
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}

df["OverallQual_Num"] = df["OverallQual"].map(quality_order)

# Trap check: Agar koi category mapping mein chhoot gayi, toh Pandas usko NaN bana dega!
# Isliye hamesha check karo:
assert df["OverallQual_Num"].isnull().sum() == df["OverallQual"].isnull().sum(), "Kuch categories map nahi huin!"
```

---

## 🟡 Level 2: One-Hot Encoding (Nominal - Jahan Koi Rank Nahi)

Agar categories ke beech koi bada-chhota nahi hai (e.g. `Gender`, `City`, `Marital_Status`, `TV_Ad_Type`):

### ⚠️ The Dummy Variable Trap (Multicollinearity Ka Baap)
Agar `Gender` column mein do values hain: `Male` aur `Female`:
Agar tum dono ke liye nayi columns bana doge:
- `is_Male` = [1, 0, 1]
- `is_Female` = [0, 1, 0]

Toh dhyaan se dekho: `is_Female = 1 - is_Male`!
Dono columns 100% perfectly correlated hain ($r = -1.0$)!
Linear Regression ke maths mein matrix inversion $(X^T X)^{-1}$ crash ho jayega ya coefficients wildly unstable ho jayenge.
Isiko **Dummy Variable Trap** bolte hain!

👉 **Rule:** Agar $k$ categories hain, toh hamesha **$k - 1$** columns banate hain! Ek column base reference ban jati hai.

### Method A: Pandas `pd.get_dummies()` (Fast EDA ke liye)
```python
# drop_first=True se k-1 columns banti hain (Dummy Trap se bachav)
# dtype=int se True/False ki jagah 1/0 milta hai
df_encoded = pd.get_dummies(df, columns=["Gender", "City"], drop_first=True, dtype=int)
```

### Method B: Scikit-Learn `OneHotEncoder` (Production & Leakage-Free Pipelines ke liye)
```python
from sklearn.preprocessing import OneHotEncoder

# drop='first': Dummy Variable Trap se bachata hai
# handle_unknown='ignore': Production mein nayi category aaye toh crash hone ki jagah 0 assign karega!
# sparse_output=False: DataFrame mein convert karne ke liye dense numpy array dega
ohe = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

encoded_features = ohe.fit_transform(df[["Gender"]])
encoded_col_names = ohe.get_feature_names_out(["Gender"])
encoded_df = pd.DataFrame(encoded_features, columns=encoded_col_names, index=df.index)
```

---

## 🔴 Level 3: High Cardinality Trap (e.g. 500 Zipcodes ya 100 Cities)

Agar tum 500 unique zipcodes par `get_dummies()` chala doge:
- Dataset mein **500 nayi columns** jud jayengi!
- Memory explode ho jayegi.
- Dataset super sparse ho jayega ("Curse of Dimensionality").

### Solution: Top-K Categories + "Other" Pattern
```python
# Step 1: Top 5 sabse frequent categories nikalo
top_5_cities = df["City"].value_counts().nlargest(5).index

# Step 2: Baaki sabko 'Other' ghoshit kar do!
df["City_Cleaned"] = df["City"].apply(lambda x: x if x in top_5_cities else "Other")

# Step 3: Ab safe tarike se One-Hot Encode karo (sirf 5 columns banengi):
df = pd.get_dummies(df, columns=["City_Cleaned"], drop_first=True, dtype=int)
```

---

## 🧭 Level 4: Feature Selection (Kaun se Columns Rakhne Hain?)

Dataset mein 100 columns ho sakti hain, par Linear Regression ko sab nahi chahiye!

### Step 1: Kooda-Kachra Columns Pehle Hi Nikalo
- **Identifier Columns**: Jaise `Id`, `Employee_ID`, `Unnamed: 0` (Yeh sirf row numbers hain, koi predictive pattern nahi rakhte!).
- **Constant Columns**: Jisme 100% rows mein ek hi value ho (Variance = 0).
- **Leakage Columns**: Jo target event ke *baad* generate hoti hain (e.g. Loan approved hone ke baad ki transaction history).

```python
# Drop useless identifier columns
cols_to_drop = ["Id", "Unnamed: 0"]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
```

### Step 2: Target ke Saath Correlation Ranking
Regression mein wahi features sabse taakatwar hote hain jinka target ke saath correlation $|r|$ high ho:
```python
# Target ke saath absolute correlation sort karo
target_corr = df.corr(numeric_only=True)["Target"].abs().sort_values(ascending=False)
print("--- Top Correlated Features with Target ---")
print(target_corr.head(10))

# Weak features (e.g. |r| < 0.05) ko drop kiya ja sakta hai
weak_features = target_corr[target_corr < 0.05].index.tolist()
```

### Step 3: Multicollinearity Detection (Feature vs Feature Correlation)
Agar do input features (jaise `TV_Budget` aur `Total_Ad_Budget` ya `Square_Feet` aur `Square_Meters`) aapas mein 95% correlated hain, toh unme se **sirf ek** ko rakhna chahiye!

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Feature correlation matrix
corr_matrix = df[selected_numeric_features].corr()

# High correlation pairs pakadna (r > 0.85)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            high_corr_pairs.append((col1, col2, round(val, 3)))

print("⚠️ Warning - High Multicollinearity Pairs (Choose only one):", high_corr_pairs)
```

---

## 🛠️ Level 5: Feature Engineering (Dimaag Laga Ke Naye Feature Banana)

Do ya teen weak features ko jod kar ek super-strong feature banaya ja sakta hai:

1. **Domain Total**:
   - `Total_SF = First_Floor_SF + Second_Floor_SF + Basement_SF`
   - `Total_Ad_Spend = TV + Radio + Newspaper`
2. **Ratios (Rates)**:
   - `Price_per_SqFt = House_Price / Total_SF`
   - `Radio_to_TV_Ratio = Radio_Spend / (TV_Spend + 1)`
3. **Age Calculation from Dates**:
   - `Building_Age = Year_Sold - Year_Built` (Year 1985 ka regression ko seedha matlab nahi pata, par "Age = 35 years" ka seedha linear relation hota hai price se!).

```python
# Feature Engineering Example
df["Building_Age"] = df["YrSold"] - df["YearBuilt"]
df["Total_Bathrooms"] = df["FullBath"] + 0.5 * df["HalfBath"]
```

---

## 🚫 The 3 Deadly Sins of Encoding (Ye Kabhi Mat Karna!)

| ❌ Deadly Sin | Kyun Gunah Hai? | Sahi Tarika ✅ |
| :--- | :--- | :--- |
| **`LabelEncoder` on Nominal Features** | `LabelEncoder` Red=0, Blue=1, Green=2 kar deta hai. Linear Regression sochega Green is double of Blue! | Use **One-Hot Encoding** (`pd.get_dummies(..., drop_first=True)`). |
| **Forgetting `drop_first=True`** | Perfect multicollinearity ho jati hai, OLS regression matrix singular ho jata hai. | Hamesha **`drop_first=True`** lagao Linear models ke liye. |
| **Encoding Before Train-Test Split** | Data leakage ho sakti hai (target encoding mein) ya test set mein nai category aane par model fail ho jayega. | Train par **`.fit()`**, Test par sirf **`.transform()`**! |

---

## 💡 Quick Recall Cheat Sheet

```python
# 1. Ordinal (Rank hai) -> Map with dict
df['Qual'] = df['Qual'].map({'Low': 0, 'Med': 1, 'High': 2})

# 2. Nominal (Rank nahi hai) -> One-Hot with drop_first
df = pd.get_dummies(df, columns=['City', 'Gender'], drop_first=True, dtype=int)

# 3. High-cardinality (>20) -> Top-N + Other
top_5 = df['City'].value_counts().nlargest(5).index
df['City'] = df['City'].apply(lambda x: x if x in top_5 else 'Other')

# 4. Multicollinearity -> Drop one if correlation between 2 features > 0.85
```

