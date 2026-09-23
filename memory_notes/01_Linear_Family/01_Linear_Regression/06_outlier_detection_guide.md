# 📖 ML Memory Notes: O - Outliers & Bivariate Analysis (Basic to Advanced)

> **Golden Law of Outliers:** 
> Outlier matlab **"Class ka woh banda jo sabse alag kaand karta hai!"** 😂 
> Linear Regression squared errors ($\sum e^2$) minimize karta hai, isliye **sirf 1 ya 2 extreme outliers poori regression line ko tedha karke barbaad kar sakte hain!**

---

## 🧭 The Outlier Decision Tree

```
                         OUTLIER DETECT HUA?
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
   DATA ENTRY / SENSOR MISTAKE?                   GENUINE EXTREME REALITY?
   (Jaise: Age = 250, Salary = -$500,             (Jaise: Luxury mansions,
   ya partial abnormal sale)                      CEO salaries, viral videos)
          │                                               │
          ▼                                       ┌───────┴───────┐
   🔥 SIDDHA DROP ROW!                            ▼               ▼
   `df = df[df['Age'] <= 100]`               CAPPING (CLIP)   LOG-TRANSFORM
                                             `np.clip()`      `np.log1p()`
```

---

## 🟢 Level 1: Visual Detection (Aankhon Se Outlier Pakadna)

Outlier do tarah ke hote hain:
1. **Univariate Outlier (Akela Number Ajeeb Hai)**: Jaise Salary = \\$50,00,000.
2. **Bivariate Outlier (Do Numbers Ka Rishta Ajeeb Hai)**: Jaise 12 saal ka bacha aur \\$1,50,000 salary! (Dono akele normal hain, par saath mein impossible!).

### Visual 1: Boxplot (Tukey's 1.5 IQR Rule)
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3))
# Jo dots whiskers (dandiyo) ke bahar dikhenge, wahi Outliers hain!
plt.boxplot(df["Salary"].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor="#2ca02c", alpha=0.6))
plt.title("Boxplot: Dots Outside Whiskers = Potential Outliers", fontsize=11, fontweight="bold")
plt.xlabel("Salary ($)")
plt.show()
```

### Visual 2: Bivariate Scatter Plot (Feature vs. Target)
```python
# Feature vs Target scatter plot karke dekho kaun sa point bheed se alag bhag raha hai:
plt.figure(figsize=(8, 4))
plt.scatter(df["Years of Experience"], df["Salary"], alpha=0.6, color="#1f77b4")
plt.title("Experience vs. Salary (Outlier Check)", fontsize=11, fontweight="bold")
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.show()
```

---

## 🟡 Level 2: Mathematical Detection (Outlier Pakadne Ke 2 Formula)

### Method 1: The IQR Rule (Industry Standard — Har Jagah Chalta Hai!)
Yeh method data ke **25% (Q1)** aur **75% (Q3)** ke beech ki doori naapta hai:

$$\text{IQR} = Q3 - Q1$$
$$\text{Lower Limit} = Q1 - 1.5 \times \text{IQR}$$
$$\text{Upper Limit} = Q3 + 1.5 \times \text{IQR}$$

```python
# IQR Method Code:
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Outlier rows filter karo
outliers = df[(df[col] < lower_limit) | (df[col] > upper_limit)]
print(f"Total Outliers in {col}: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
print(f"Acceptable Range: [{lower_limit:.2f}, {upper_limit:.2f}]")
```

---

### Method 2: The Z-Score Rule (Sirf Normal/Bell-Curve Data Ke Liye)
Agar data pehle se Bell-Curve hai, toh **68-95-99.7 Rule** lagta hai:
- 99.7% data mean se **3 Standard Deviations ($\pm 3\sigma$)** ke andar hota hai.
- Jo bhi point $\pm 3\sigma$ ke bahar gaya $\to$ Outlier!

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df[col].dropna()))
outliers_z = df[z_scores > 3]
print(f"Z-Score Outliers (>3 Std Dev): {len(outliers_z)}")
```

---

## 🔴 Level 3: Outliers Ka Ilaaj (Handling Strategies)

Pakad liya outlier, ab karein kya? **4 Raaste** hote hain:

### 1. Trimming / Dropping (Fek Do)
Agar data entry error hai (jaise Age = 200 ya negative income):
```python
df_clean = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)].copy()
```

### 2. Capping / Winsorization (Clamping — Sir Pe Danda Maaro!)
Data row ko delete mat karo! Agar koi value Upper Limit se badi hai, toh usko Upper Limit par hi rok do:
```python
# Jo bada hai usko upper limit bana do, jo chota hai usko lower limit:
df[col] = np.clip(df[col], lower_limit, upper_limit)
```
💡 **Fayda:** Data points lose nahi hote, aur extreme values model ko kharab nahi karti!

### 3. Log Transformation
Badi values ko shrink kar deta hai (`np.log1p(df[col])`).

### 4. Robust Regression (Model Badal Do)
Agar outliers genuine business reality hain (jaise Fraud detection ya Stock Market), toh Linear Regression ki jagah **Ridge / Huber Regressor** ya **Tree Models (Random Forest)** use karo jo outliers se darrte nahi hain!

---

## 🟣 Level 4: Bivariate & Correlation Analysis (Part of O)

Outlier dekhne ke baad hum dekhte hain: **Kaun sa Feature Target Ka Sabse Bada Dost Hai?**

### 1. Pearson Correlation Coefficient ($r$):
- **$+1.0$**: Perfect Dosti (Jaise jaise X badhega, y bhi badhega).
- **$0.0$**: Anjaan (Koi lena-dena nahi).
- **$-1.0$**: Dushmani (X badhega toh y ghatega).

```python
# Sabhi numeric columns ka Target ke saath rishta:
corr_matrix = df.select_dtypes(include=np.number).corr()
print("Top Correlated Features with Target:")
print(corr_matrix[target_col].sort_values(ascending=False))
```

### 2. Correlation Heatmap
```python
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title("Feature Correlation Matrix", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 📋 The "Copy-Paste" Template for Any Project

```python
# --- STEP 2.5: O - OUTLIERS & CORRELATION AUDIT ---

# 1. IQR Outlier Check on Target
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_count = ((df[target] < lower) | (df[target] > upper)).sum()
print(f"Target '{target}' Outliers (IQR): {outliers_count} ({outliers_count/len(df)*100:.1f}%)")

# 2. Correlation Ranking
numeric_df = df.select_dtypes(include=[np.number])
if target in numeric_df.columns:
    print("\n--- Correlation with Target ---")
    print(numeric_df.corr()[target].sort_values(ascending=False).round(3))
```

