# 🌐 Common Linear Family: 03 - Feature Segregation & Cardinality Rules

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho agar computer ko bolo: *"Formula lagao: $2 \times \text{'Blue'} + 5$"*.  
> Computer bolega: *"Bhai, pagal ho kya? Blue ko 2 se multiply kaise karoon?"*  
> Linear models sirf numbers samajhte hain: $\mathbf{w}^T \mathbf{x} + b$.  
> Isliye data aate hi usko 3 balti (buckets) mein baantna padta hai:  
> 1. **Numbers (Continuous)** $\to$ Inhe scale karna hai.  
> 2. **Text / Categories** $\to$ Inhe 0 aur 1 (One-Hot) banana hai.  
> 3. **ID Columns** $\to$ Inhe dustbin mein phenkna hai!

---

## 🧭 The 3-Bucket Segregation Rule

```
                            RAW DATASET COLUMNS
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
 CONTINUOUS NUMERICAL       CATEGORICAL / TEXT          DROP CANDIDATES
 - Area, Income, Price      - Neighborhood, Color       - Id, Patient_ID, RollNo
 - Treated with Scaling     - Treated with One-Hot      - Pure Noise / Leakage
 - StandardScaler()         - OneHotEncoder()           - df.drop(columns=[...])
```

---

## 🧠 The Cardinality Threshold Trick (Senior Developer Secret)

Kabhi-kabhi dataset mein columns hote hain jaise `OverallQual` (Values: 1 to 10) ya `EducationLevel` (Values: 1, 2, 3).  
Technically Pandas inko `int64` (number) bolega, lekin asal mein yeh **Categories** hain!

Isliye hum ek rule lagate hain:
$$\text{If } \text{dtype} == \text{object} \quad \text{OR} \quad \text{nunique()} \le 15 \implies \text{CATEGORICAL}$$

Isse koi bhi disguised category galat numeric bucket mein nahi jaati!

---

## 📋 Copy-Paste Boilerplate: Automated 3-Bucket Feature Segregator
*(Isko copy karke kisi bhi notebook ke Step 3 mein paste karo)*

```python
# ==============================================================================
# 🌐 STEP 3: AUTOMATED 3-BUCKET FEATURE SEGREGATOR
# ==============================================================================
import pandas as pd

def segregate_features(df, target_col, drop_cols=None, cardinality_threshold=15):
    """
    Automatically classifies columns into:
    1. Continuous features (Numeric and nunique > cardinality_threshold)
    2. Categorical features (Object or nunique <= cardinality_threshold)
    3. Drop features (ID, target, or specified leakage columns)
    """
    if drop_cols is None:
        drop_cols = []
        
    # Auto-detect ID columns if not specified
    for col in df.columns:
        if col.lower() in ['id', 'id_col', 'user_id', 'customer_id', 'roll_no']:
            if col not in drop_cols:
                drop_cols.append(col)

    continuous_features = []
    categorical_features = []
    
    for col in df.columns:
        if col == target_col or col in drop_cols:
            continue
            
        n_unique = df[col].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Rule: Object/Text OR Low-Cardinality Number -> Categorical
        if not is_numeric or n_unique <= cardinality_threshold:
            categorical_features.append(col)
        else:
            continuous_features.append(col)

    print("=" * 70)
    print("🔍 FEATURE SEGREGATION REPORT")
    print("=" * 70)
    print(f"Target Column          : '{target_col}'")
    print(f"Continuous Features    : {len(continuous_features)} cols -> (Will apply StandardScaler)")
    print(f"Categorical Features   : {len(categorical_features)} cols -> (Will apply OneHotEncoder)")
    print(f"Dropped ID / Artifacts : {len(drop_cols)} cols -> {drop_cols}")
    print("=" * 70)
    
    return continuous_features, categorical_features, drop_cols

# Usage Example:
# cont_cols, cat_cols, drop_cols = segregate_features(df, target_col='SalePrice', drop_cols=['Id'])
```

