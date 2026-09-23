# 🎯 Classification Memory Notes: 05 - Target Variable Analysis & Zero-Rule Baseline

> **The Senior Developer's Golden Law of Target Analysis:**
> Regression mein hum poochte the *"Kitna?"* (Continuous Value - jaise House Price, Salary) $\implies$ Skewness aur Normal Distribution dekhte the.
> Classification mein hum poochte hain *"Kaunsa?"* ya *"Haan ya Na?"* (Discrete Class - jaise Spam vs Ham, Disease vs Normal, Approved vs Denied) $\implies$ Yahan **Class Balance & Proportions** dekhi jaati hai!
> 
> **The Baseline Rule:** 
> Hamesha **Majority Class Baseline Accuracy** calculate karo. Agar aapka ML model baseline se zyada accuracy nahi de raha, toh model ne data se kuch bhi naya nahi seekha!

---

## 🧭 The Universal Classification Target Decision Tree

```
                           TARGET VARIABLE (y)
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
BINARY CLASSIFICATION                             MULTICLASS CLASSIFICATION
Exactly 2 Classes (0 or 1)                        3 ya usse zyada Classes
- Denied (0) vs Approved (1)                      - Sentiment: Neg (0), Neu (1), Pos (2)
- Healthy (0) vs Diseased (1)                     - Iris: Setosa (0), Versicolor (1), Virginica (2)
- Normal (0) vs Fraud (1)                         - Rating: Low (0), Medium (1), High (2)
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ▼
              CLASS BALANCE AUDIT (`value_counts(normalize=True)`)
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
  BALANCED DATASET                                  IMBALANCED DATASET
  Ratio: 50:50, 55:45, 60:40                        Ratio: 80:20, 90:10, 99:1
  - Accuracy metric reliable hai                    - ⚠️ THE ACCURACY TRAP! Accuracy dhokha degi!
  - Standard Train-Test Split OK                    - Precision, Recall, F1, ROC-AUC dekhna zaroori!
                                                    - `class_weight='balanced'` ya SMOTE zaroori!
```

---

## 🔬 Section 1: Standard 0 vs 1 Binary Convention

Machine learning models text labels (`'No'/'Yes'`, `'Denied'/'Approved'`, `'Ham'/'Spam'`) directly nahi samajhte. Hamesha binary integer mapping use karein:
* **Class 0 (Negative / Baseline State):** Normal, Healthy, Legitimate, Denied, Ham.
* **Class 1 (Positive / Event of Interest):** Fraud, Diseased, Approved, Spam, Churn.

```python
import pandas as pd
import numpy as np

# Universal Target Column Variable
target_col = 'target'  # Apne dataset ke target column ka naam yahan likhein

# Agar target text format mein hai, toh integer 0 aur 1 mein map karein:
# Example: df[target_col] = df[target_col].map({'Denied': 0, 'Approved': 1})

# Sanity Check: Verify karo ki target mein sirf 0 aur 1 hain
print(f"Target '{target_col}' Unique Values : {df[target_col].unique()}")
assert set(df[target_col].unique()).issubset({0, 1}), "❌ Target must only contain 0 and 1!"
```

---

## 📊 Section 2: Target Audit (Counts & Percentage Breakdown)

```python
# 1. Exact Headcount
counts = df[target_col].value_counts(dropna=False)

# 2. Percentage Proportions
proportions = df[target_col].value_counts(normalize=True, dropna=False) * 100

summary_target = pd.DataFrame({
    'Headcount': counts,
    'Percentage (%)': proportions.round(2)
})

print("=" * 55)
print(f"🎯 TARGET VARIABLE AUDIT: '{target_col}'")
print("=" * 55)
print(summary_target)

# Imbalance Ratio calculate karo
majority_pct = proportions.max()
minority_pct = proportions.min()
imbalance_ratio = majority_pct / minority_pct
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3.0:
    print("⚠️ Status: IMBALANCED DATASET! Accuracy dhokha de sakti hai. Use F1 / ROC-AUC.")
else:
    print("✅ Status: WELL BALANCED! Accuracy is a reliable primary metric.")
```

---

## 📈 Section 3: Visualizing Target Distribution (Bar Chart)

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
palette = ['#e74c3c', '#27ae60']  # Red for Class 0, Green for Class 1

ax = sns.countplot(data=df, x=target_col, hue=target_col, palette=palette, legend=False)
plt.title(f"Target Distribution: '{target_col}' (Class 0 vs Class 1)", fontsize=11, fontweight='bold')
plt.xlabel(f"{target_col} (0 = Negative / Denied, 1 = Positive / Approved)")
plt.ylabel("Number of Samples")

# Bars ke upar exact number aur percentage likho
total = len(df)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        pct = f"{100 * height / total:.1f}%"
        ax.annotate(f"{int(height):,}\n({pct})",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()
```

---

## ⚠️ Section 4: The Accuracy Trap (Kyu 99% Accuracy Dhokha Ho Sakti Hai?)

Sochiye ek Fraud Detection model banaya gaya:
* Total Transactions: 10,000
* Legitimate Transactions (Class 0): 9,900 (99%)
* Fraud Transactions (Class 1): 100 (1%)

Agar aapka model bilkul dimaag na lagaye aur har transaction ko **"Legitimate" (0)** bol de:
$$\text{Accuracy} = \frac{9,900}{10,000} = \mathbf{99\%!}$$

Model ne 99% accuracy claim ki, lekin haqeeqat mein **100 ke 100 frauds miss ho gaye aur bank loot gaya!** Model ne 0% fraud detect kiya.

> [!CAUTION]
> **The Golden Trap Rule:**
> Jab bhi dataset **Imbalanced** ho (jaise Fraud 1%, Rare Disease 2%), **Accuracy dekhna sabse bada trap hai!** 
> Hamesha **Confusion Matrix, Precision, Recall, F1-Score aur ROC-AUC** check karna padta hai!

---

## ⚖️ Section 5: Establishing Zero-Rule Baseline (DummyClassifier)

Kissi bhi classification model ko evaluate karne se pehle humein **Baseline Benchmark** pata hona chahiye:
Agar model koi feature na dekhe aur sirf Majority Class guess kare, toh kitni accuracy aayegi?

```python
from sklearn.dummy import DummyClassifier

# 1. Feature Matrix (X) aur Target Vector (y) alag karo
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Dummy Classifier (Most frequent class guess karne wala)
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X, y)

# 3. Baseline Accuracy Score nikaalo
baseline_acc = dummy.score(X, y) * 100

print("=" * 60)
print(f"📊 ZERO-RULE BASELINE ACCURACY: {baseline_acc:.2f}%")
print("=" * 60)
print("🎯 Target Class Distribution (%):")
print((y.value_counts(normalize=True) * 100).round(2).to_dict())
print("-" * 60)
print(f"👉 Model Benchmark: Logistic Regression ko is {baseline_acc:.2f}% baseline score se behtar perform karna hoga!")
print("=" * 60)
```

---

## 🧠 Universal Interview Flashcard: Target Analysis

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Classification mein target variable analysis kaise karte ho?"** | Mean aur standard deviation nikalta hoon. | Classification mein target continuous nahi hai! Frequency counts (`value_counts`), Class proportions (%), aur Imbalance ratio dekha jata hai. |
| **"Agar model ki accuracy 95% aayi toh kya model accha hai?"** | Haan, 95% toh bohot accha score hai! | Pehle majority class baseline dekhna padega! Agar class 0 already 95% thi, toh model ne kuch nahi seekha (The Accuracy Trap). |
| **"Baseline accuracy kya hoti hai?"** | Hamesha 50% hoti hai. | Hamesha 50% nahi hoti! Woh dataset ki majority class proportion hoti hai jo `DummyClassifier(strategy='most_frequent')` achieve karta hai. |
