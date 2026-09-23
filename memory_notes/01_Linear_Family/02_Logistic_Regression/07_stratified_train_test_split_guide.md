# 🎯 Classification Memory Notes: 07 - Stratified Train-Test Splitting (Zero-Leakage Architecture)

> **The Golden Law of Machine Learning:**
> Data Preprocessing (Scaling ya Encoding) **Train-Test Split se pehle poore dataset par karna ILLEGAL hai!**
> Agar aapne poore dataset par pehle StandardScaler ya OneHotEncoder lagaya, toh Test set ki information (mean, std dev, category boundaries) training mein leak ho jayegi (**Data Leakage**).
> Isliye: **Pehle Data ko Baanto (Split), Fir Transformation Seekho (Fit)!**

---

## 🧭 The Zero-Leakage Splitting Architecture

```
                               RAW DATASET (df)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     FEATURE MATRIX (X)                              TARGET VECTOR (y)
   df.drop(columns=[target_col])                     df[target_col]
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      ▼
                        STRATIFIED TRAIN-TEST SPLIT
                        `stratify=y` (e.g., 80% Train, 20% Test)
                                      │
         ┌────────────────────────────┴────────────────────────────┐
         ▼                                                         ▼
  TRAIN SET (80%)                                           TEST SET (20%)
  X_train, y_train                                          X_test, y_test
  (Model isse padhega aur seekhega)                         (Strictly Unseen - Final Exam)
         │                                                         │
         ▼                                                         ▼
  FIT & TRANSFORM                                           TRANSFORM ONLY
  (Learn Mean, Std, Categories)                             (Use Train's learned rules)
```

---

## 🚫 Why Simple Random Split Fails for Classification

### 1. The Class Vanishing Trap
Sochiye aapke dataset mein **1,000 applicants** hain jisme:
- **Class 0 (Normal / Denied):** 950 records (95%)
- **Class 1 (Fraud / Approved):** 50 records (5%)

Agar aapne bina `stratify=y` ke simple random split kiya:
* Pure random luck se 800 train records mein **48 Fraud** chale gaye.
* Aur 200 test records mein sirf **2 Fraud** bache (1%)!
* **Nateeja:** Test set ka class balance bigad gaya. Model test set par fraud detect hi nahi kar payega aur evaluation invalid ho jayegi!

### 2. The Solution: `stratify=y` (Proportional Sampling)
`stratify=y` Scikit-Learn ko bolta hai:
> *"Train aur Test dono tukdo mein Class 0 aur Class 1 ka ratio exactly wahi hona chahiye jo original dataset mein tha!"*

| Set | Total Rows | Class 0 (95%) | Class 1 (5%) |
| :--- | :---: | :---: | :---: |
| **Full Dataset** | 1,000 | 950 (95.0%) | 50 (5.0%) |
| **Train Set (80%)** | 800 | 760 (95.0%) | 40 (5.0%) |
| **Test Set (20%)** | 200 | 190 (95.0%) | 10 (5.0%) |

---

## 💻 Production Implementation: Stratified Train-Test Split

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Feature Matrix (X) aur Target Vector (y) ko separate karo
target_col = 'target'  # Apne target column ka naam yahan likhein

X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Stratified Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20,       # 20% data testing (exam) ke liye alag rakhein
    random_state=42,      # Reproducibility ke liye seed
    stratify=y            # NON-NEGOTIABLE: Class proportions lock karta hai
)

# 3. Shapes Summary Print karo
print("=" * 60)
print("📦 STRATIFIED TRAIN-TEST SPLIT SUMMARY")
print("=" * 60)
print(f"Original Data Shape : {df.shape}")
print(f"X_train Shape       : {X_train.shape} (80% learning data)")
print(f"X_test Shape        : {X_test.shape}  (20% unseen test data)")
print(f"y_train Shape       : {y_train.shape}")
print(f"y_test Shape        : {y_test.shape}")

# 4. Stratification Sanity Check (Verification)
train_balance = (y_train.value_counts(normalize=True) * 100).round(2).to_dict()
test_balance  = (y_test.value_counts(normalize=True) * 100).round(2).to_dict()

print("-" * 60)
print("🎯 Class Distribution Balance Verification:")
print(f"Train Set Balance (%) : {train_balance}")
print(f"Test Set Balance (%)  : {test_balance}")
print("=" * 60)
```

---

## 🛡️ Special Splitting Traps & Senior Edge Cases

### Trap 1: Time-Series / Temporal Data mein `stratify=y` lagana
- **Galti:** Agar data time-dependent hai (jaise Stock Market, Monthly Sales, ya User Activity over time), toh `train_test_split` ya `stratify` **kabhi mat lagao**!
- **Kyun:** Shuffling se future ka data past mein chala jayega (Lookahead Bias).
- **Fix:** Hamesha time ke hisaab se date cut-off se split karo (e.g., Jan–Oct $\to$ Train, Nov–Dec $\to$ Test) ya `TimeSeriesSplit` use karo.

### Trap 2: Repeated Patients / Grouped Data (Patient Level Leakage)
- **Galti:** Agar ek hi patient ke 5 alag-alag visits ki rows hain, aur random split kiya:
  * Patient ki 3 visits Train mein aur 2 visits Test mein chali jayengi.
  * Model patient ke unique habits ya biology ko ratta maar lega!
- **Fix:** `from sklearn.model_selection import GroupShuffleSplit` use karo taaki ek patient ki saari rows ya toh Train mein hon ya Test mein!

---

## 🧠 Universal Interview Flashcard: Train-Test Splitting

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Train-Test Split kab karna chahiye?"** | Sabhi features scale aur encode karne ke baad. | **Hamesha Preprocessing se pehle!** Taaki test data ki koi bhi statistic (mean, std, categories) training mein leak na ho. |
| **"Classification mein `stratify=y` kyu zaroori hai?"** | Acche results ke liye. | Random sampling se test set mein rare/minority class ka proportion badal sakta hai ya zero ho sakta hai. Stratification dono tukdo mein class distribution barabar rakhta hai. |
| **"`random_state=42` ka kya kaam hai?\"** | Accuracy badhata hai. | Accuracy par koi asar nahi padta; yeh sirf random number generator ko seed deta hai taaki code dobara chalane par wahi identical split mile (Reproducibility). |
| **"Agar dataset bohot chhota ho (jaise 100 rows)?"** | 50:50 split kar do. | `StratifiedKFold(n_splits=5)` Cross-Validation use karo taaki har data point test aur train dono mein use ho sake. |
