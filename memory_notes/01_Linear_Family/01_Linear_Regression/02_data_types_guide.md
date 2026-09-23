# 📖 ML Memory Notes: D - Data Types Check (Basic to Advanced)

> **Golden Rule of Data Types:** 
> Machine Learning models sirf **Numbers (floats & integers)** samajhte hain. Agar kisi number column ko Python ne galti se **Object (Text)** samajh liya, toh model uspar train hi nahi ho payega!

---

## 🟢 Level 1: Basic Inspection (Kundali Dekhna)

### 1. Simple Data Types Check
```python
# Har column ka type dekho
print(df.dtypes)

# Summary count dekho: Kitne columns number hain aur kitne text hain?
print(df.dtypes.value_counts())
```

### 2. The All-in-One Inspection (`df.info()`)
Sabse powerful single command jo Data Types + Missing Values dono ek saath dikhati hai:
```python
df.info()
```
💡 **Kya dekhna hota hai?**
- `int64`: Poore numbers (e.g. 1, 2, 50, 100)
- `float64`: Decimal numbers (e.g. 12.5, 3.14, NaN)
- `object`: Text / Strings ya Mixed data
- `bool`: True / False

---

## 🟡 Level 2: Intermediate (Numbers Aur Text Ko Alag Karna)

Kabhi bhi hath se column ke naam mat chuno! Pandas ka **`select_dtypes`** automatic filter karta hai:

```python
import numpy as np

# 1. Saare NUMERICAL columns alag nikalo:
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 2. Saare CATEGORICAL / TEXT columns alag nikalo:
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numbers wale columns ({len(numeric_cols)}):   {numeric_cols}")
print(f"Text wale columns    ({len(categorical_cols)}): {categorical_cols}")
```

💡 **Fayda:** Kal ko dataset mein 100 columns bhi hon, yeh 2 lines bina kisi galti ke sabko alag kar dengi!

---

## 🔴 Level 3: Advanced (Real-World Data Type Traps & Fixes)

Asli datasets mein data types hamesha clean nahi aate. Yahan 3 sabse bade traps aate hain:

### ⚠️ Trap 1: "Fake String" (Number hai, lekin Python usko Text bol raha hai!)
**Kyu hota hai?** 
Data mein `$`, `,`, ya `%` ka symbol laga hota hai (jaise: `"$90,000"` ya `"50%"`). 
Python symbol dekh kar confuse ho jata hai aur poore column ko `object` bana deta hai!

**Solution (The 2-Step Cleaner):**
```python
# 1. Faltu symbols hatao
df["Salary"] = df["Salary"].astype(str).str.replace("$", "", regex=False)
df["Salary"] = df["Salary"].str.replace(",", "", regex=False)

# 2. Number mein badal do (Float)
df["Salary"] = df["Salary"].astype(float)
```

---

### ⚠️ Trap 2: Ajeeb Garbage Data (`pd.to_numeric` with `errors='coerce'`)
Agar kisi number column ke beech mein kisi ne `"Unknown"` ya `"?"` likh diya ho, toh `.astype(float)` crash ho jayega.

**Weapon: `errors='coerce'`**
```python
# Jo number ban sakta hai banega, jo garbage hai woh chup-chaap NaN ban jayega!
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
```

---

### ⚠️ Trap 3: "Fake Number" (Text/Category hai, lekin Python usko Number bol raha hai!)
**Kyu hota hai?**
- `PClass`: 1st, 2nd, 3rd class (Ticket class)
- `ZipCode`: 90210, 110001 (Pincode)
- `Month`: 1, 2, 3... 12

Python sochega: *"Arey 90210 toh 110001 se chota number hai!"* 
Model galat math laga dega!

**Solution:** Inhe wapas string/category banao:
```python
df["ZipCode"] = df["ZipCode"].astype(str)
```

---

### ⚠️ Trap 4: Date-Time Columns
Agar date `"2024-05-15"` likha hai, toh Python usko normal text samajhta hai.

**Solution:**
```python
# String se asli DateTime banao
df["Date"] = pd.to_datetime(df["Date"])

# Ab isme se saal, mahina, din nikal sakte ho:
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
```

---

## 🟣 Level 4: Pro Optimization (RAM Bachana)

Bade datasets (10 lakh+ rows) mein:
- Text columns ko **`category`** type banane se RAM usage **80% kam** ho jaati hai!
```python
# RAM bachane ke liye:
df["Gender"] = df["Gender"].astype("category")
```

---

## 📋 The "Copy-Paste" Template for Any Project

Har naye project mein yeh cell hamesha run karo:

```python
# --- STEP 2.1: D - DATA TYPES AUDIT ---

# 1. Overview count
print("--- Data Types Count ---")
print(df.dtypes.value_counts())

# 2. Auto-separate numbers and text
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"\nNumerical Columns ({len(num_cols)}):   {num_cols}")
print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")

# 3. Sanity check for fake numbers (like ID)
id_like = [c for c in num_cols if "id" in c.lower()]
if id_like:
    print(
        f"\n⚠️ Alert: Yeh columns ID lag rahe hain, model mein mat daalna: {id_like}"
    )
```
