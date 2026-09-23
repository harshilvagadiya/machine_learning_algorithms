# 📖 ML Memory Notes: T - Target Variable Analysis (Basic to Advanced)

> **Golden Law of the Target ($y$):** 
> Poore Machine Learning project ka **Hero** tumhara Target variable ($y$) hai! Baaki saare columns ($X$) sirf is ek target ko predict karne ke liye jeete hain. Agar tumne target ka swabhav (distribution) nahi samjha, toh model kabhi sahi prediction nahi dega!

---

## 🧭 The Target Analysis Decision Tree

```
                         TARGET VARIABLE (y)
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
CONTINUOUS NUMBER? (Regression)                 CATEGORY / CLASS? (Classification)
(Jaise: House Price, Salary, Sales)             (Jaise: Pass/Fail, Churn/Retain)
        │                                               │
        ▼                                               ▼
1. Skewness Check (`.skew()`)                   1. Class Balance Check (`.value_counts()`)
2. Distribution (Histogram + Q-Q Plot)          2. Bar Chart of Class Ratios
3. Outliers & Min/Max Check                     3. Imbalance Trap Check (95% vs 5%?)
        │
        ▼
Is it heavily Right-Skewed? (Tail on right)
        │
   ┌────┴────┐
   ▼         ▼
  YES        NO (Already Bell Curve)
   │         │
   ▼         ▼
Log-Transform! Keep As-Is!
`np.log1p(y)`
```

---

## 🟢 Level 1: Basic Inspection (Summary & Sanity Check)

### 1. The 5-Point Summary (`df[target].describe()`)
```python
# Target ki poori kundali dekho
print(df[target_col].describe().round(2))
```
💡 **Kya dekhna hota hai?**
- **Min**: Kya koi impossible number hai? (Jaise negative salary `-$500` ya negative house price?).
- **Mean vs. Median (50%)**:
  - Agar `Mean ≈ Median` $\to$ Data **Symmetric (Normal)** hai!
  - Agar `Mean > Median` $\to$ Data **Right-Skewed** hai (kuch ameer log average ko oopar kheench rahe hain!).

### 2. Missing Target Check (Strict Rule!)
```python
null_target = df[target_col].isnull().sum()
if null_target > 0:
    print(f"⚠️ DANGER: {null_target} rows mein Target gayab hai! Siddha drop karo:")
    df = df.dropna(subset=[target_col]).copy()
```

---

## 🟡 Level 2: Visualizing the Target (The 3 Core Plots)

Linear Regression assumptions check karne ke liye hum yeh 3 plots banate hain:

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 1. Histogram with KDE (Distribution shape)
axes[0].hist(df[target_col], bins=30, color="#1f77b4", edgecolor="black", alpha=0.75, density=True)
kde_x = np.linspace(df[target_col].min(), df[target_col].max(), 200)
kde = stats.gaussian_kde(df[target_col].dropna())
axes[0].plot(kde_x, kde(kde_x), color="#d62728", lw=2, label="KDE Density")
axes[0].set_title(f"Histogram (Skewness: {df[target_col].skew():.2f})", fontsize=11, fontweight="bold")
axes[0].set_xlabel(target_col)
axes[0].legend()

# 2. Boxplot (Outliers check)
axes[1].boxplot(df[target_col].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#2ca02c", alpha=0.6))
axes[1].set_title("Boxplot (Spread & Outliers)", fontsize=11, fontweight="bold")
axes[1].set_xlabel(target_col)

# 3. Normal Q-Q Plot (Kya data Normal Bell Curve ko follow karta hai?)
stats.probplot(df[target_col].dropna(), dist="norm", plot=axes[2])
axes[2].set_title("Normal Q-Q Plot", fontsize=11, fontweight="bold")
axes[2].get_lines()[0].set_color("#1f77b4")
axes[2].get_lines()[1].set_color("#d62728")

plt.tight_layout()
plt.show()
```

---

## 🔴 Level 3: Skewness (Kyu Linear Regression Isse Pareshan Hota Hai?)

### What is Skewness?
- **Skewness $\approx 0$**: Bell-shaped normal curve (Subhanallah! Ekdum perfect!).
- **Skewness $> +0.75$ ya $+1.0$ (Right-Skewed)**:
  - Zyadatar log left mein baithe hain, aur right mein ek lambi poonch (tail) hai (jaise: Ames House Price, Salary).
  - **Problem**: Linear Regression seedhi line kheenchta hai. Ameer luxury houses line ko zabardasti oopar kheench lete hain, jisse aam gharon ka prediction kharab ho jata hai!

---

## 🟣 Level 4: Advanced — Log Transformation (Skewness Ka Ilaaj!)

Jab target bohot right-skewed ho, toh hum mathematical magic lagate hain: **Logarithm ($\log$)**!

### 1. Transformation Apply Karo (`np.log1p`):
`np.log1p(y)` ka matlab hota hai: $\log(1 + y)$ (1 isliye jodte hain taaki agar zero ho toh error na aaye).

```python
# Target ko normal bell curve mein badal do:
y_log = np.log1p(df[target_col])

print(f"Purana Skewness: {df[target_col].skew():.2f}")
print(f"Naya Skewness:   {y_log.skew():.2f} (Clean Normal Curve!)")
```

### 2. Predictions Ko Wapas Asli Dollars Mein Laana (`np.expm1`):
Model log values predict karega (jaise 11.5 ya 12.1). Client ko bolenge toh bolega *"11 dollar salary?!"*
Isliye prediction ke baad **Reverse (Inverse)** karna padta hai:

```python
# Log se wapas asli dollars banao:
y_pred_dollars = np.expm1(y_pred_log)
```

💡 **Short Trick:**
- **Padhai se pehle:** `np.log1p(y)`
- **Prediction ke baad:** `np.expm1(pred)`

---

## 📋 The "Copy-Paste" Template for Any Project

```python
# --- STEP 2.4: T - TARGET VARIABLE ANALYSIS ---

target = "Sales ($)"  # Apne target ka naam likho

print("--- Target Summary Statistics ---")
print(df[target].describe().round(2))

skew_val = df[target].skew()
print(f"\nTarget Skewness: {skew_val:.2f}")

if abs(skew_val) < 0.5:
    print("✅ Distribution: Fairly Symmetric (Normal Curve ke kareeb hai).")
elif skew_val > 0.5:
    print("⚠️ Distribution: Right-Skewed (Consider Log-Transformation if skew > 1.0).")
else:
    print("⚠️ Distribution: Left-Skewed.")
```

