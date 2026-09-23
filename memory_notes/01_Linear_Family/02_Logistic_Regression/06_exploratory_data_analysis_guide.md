# 🎯 Classification Memory Notes: 06 - Exploratory Data Analysis (EDA) & Feature Relationships

> **The Senior Developer's Golden Rule for Classification EDA:**
> Regression mein hum $x$ aur $y$ ka seedha Scatter Plot aur Pearson Correlation dekh lete the kyunki target continuous number tha.
> Lekin **Classification** mein target discrete binary class ($0$ ya $1$) hota hai! Agar scatter plot banaoge, toh do parallel horizontal lines dikhengi jo kisi kaam ki nahi hain.
> Yahan humein **Group Separation (Overlap vs. Distinction)**, **Statistical Divergence ($\chi^2$)**, aur **Odds Shifts** dekhne hote hain!
> Aur sabse zaroori: Hum Step 02 mein banaye gaye **`continuous_features`** aur **`categorical_features`** ko hi seedha use karte hain!

---

## 🧭 The Universal Classification EDA Framework (For ANY Dataset)

```
                            FEATURE ANALYSIS wrt TARGET (y)
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
 CONTINUOUS FEATURES vs TARGET                               CATEGORICAL FEATURES vs TARGET
 continuous_features (Age, Income, Balance)                  categorical_features (Job, State, Tier)
         │                                                                   │
         ▼                                                                   ▼
 1. Grouped Summary Statistics (`df.groupby(target_col)`)     1. Cross-Tabulation (`pd.crosstab`)
 2. Overlaid KDE / Density Histograms                        2. 100% Stacked Bar Charts (Prevalence)
 3. Boxplots (Median separation & Outliers)                  3. Chi-Square Test of Independence ($\chi^2$)
 4. Point-Biserial Correlation Ranking                       4. Information Value (IV) / Odds Ratios
```

---

## 🟢 Section 1: Continuous Features vs Target

### 1. Grouped Summary Statistics
Dono classes ke beech har continuous feature ka Mean aur Median compare karein:

```python
# Step 02 mein identify kiye gaye continuous_features use karein:
grouped_stats = df.groupby(target_col)[continuous_features].agg(['mean', 'median', 'std']).round(2)

print("=" * 65)
print("📊 GROUPED SUMMARY STATISTICS BY TARGET")
print("=" * 65)
print(grouped_stats.T)
```

💡 **The Separation Test (Kya dekhna hai?):**
* Agar **Class 0** aur **Class 1** ke medians mein zameen-aasmaan ka farq hai (jaise Approved Debt = 1.2 vs Denied Debt = 4.8) $\implies$ **High Discriminating Power!** Model is feature se aasani se separate karega.
* Agar dono classes ka distribution ek doosre ke upar 90% overlap kar raha hai $\implies$ **Weak Feature** (Linear model akela isse separate nahi kar sakta).

### 2. Boxplots (Median Separation & Outliers)
```python
import matplotlib.pyplot as plt
import seaborn as sns

cols_to_plot = continuous_features[:4]

if cols_to_plot:
    fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(4.5 * len(cols_to_plot), 4))
    if len(cols_to_plot) == 1:
        axes = [axes]
    
    palette = ['#e74c3c', '#27ae60']  # Red for Class 0, Green for Class 1
    for idx, col in enumerate(cols_to_plot):
        sns.boxplot(data=df, x=target_col, y=col, hue=target_col, palette=palette, legend=False, ax=axes[idx], width=0.4)
        axes[idx].set_title(f"{col} by {target_col}", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel(f"{target_col} (0 = Negative, 1 = Positive)")
        axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
```

---

## 🟡 Section 2: Categorical Features vs Target ($\chi^2$ Independence Test)

Categorical predictors (Gender, Payment Method, Pain Type, Account Tier) ka target ke saath rishta statistically significant hai ya nahi:

### 1. Cross-Tabulation & 100% Proportions
```python
# Har category ke andar target class ka percentage
for col in categorical_features[:3]:
    ct_norm = pd.crosstab(df[col], df[target_col], normalize='index') * 100
    print(f"\n--- 📑 Cross-Tabulation for '{col}' (%) ---")
    print(ct_norm.round(2))
```

### 2. Chi-Square Test of Independence ($\chi^2$)
* **Null Hypothesis ($H_0$):** Feature aur Target ke beech koi statistical rishta nahi hai (Independent).
* **Alternative Hypothesis ($H_1$):** Feature aur Target strongly associated hain ($p < 0.05$).

```python
from scipy.stats import chi2_contingency

chi2_results = []
for col in categorical_features:
    ct = pd.crosstab(df[col], df[target_col])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    chi2_results.append({
        'Feature': col,
        'Chi2_Stat': round(chi2, 2),
        'p_value': f"{p_val:.2e}",
        'Significant': '✅ Yes' if p_val < 0.05 else '❌ No'
    })

chi2_df = pd.DataFrame(chi2_results).sort_values(by='Chi2_Stat', ascending=False)
print("=" * 55)
print("🔬 CHI-SQUARE TEST OF INDEPENDENCE")
print("=" * 55)
print(chi2_df.to_string(index=False))
```

---

## 🔴 Section 3: Point-Biserial Correlation Diagnostics

Continuous features aur binary target ($0/1$) ke beech linear correlation calculate karke rank karein:

```python
# Compute and rank linear correlation with target
corr_ranking = df[continuous_features + [target_col]].corr()[target_col].drop(target_col).sort_values(ascending=False)

print("=" * 55)
print("🏆 RANKED CORRELATION WITH TARGET")
print("=" * 55)
print(corr_ranking.round(3))
```

---

## 📋 Section 4: Universal Plug & Play Classification EDA Engine

Is function ko kisi bhi naye classification project mein run karo — yeh continuous boxplots aur categorical stacked proportion plots ek saath render karta hai (modern Pandas & Seaborn compatible):

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def universal_classification_eda(df, target_col, max_num_cols=5, max_cat_cols=4):
    """
    Renders universal visual EDA for any classification dataset.
    Compatible with modern Pandas StringDtype and Seaborn palettes.
    """
    X = df.drop(columns=[target_col], errors='ignore')
    
    # Safe numerical check (Supports modern Pandas StringDtype without crashing)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 10][:max_num_cols]
    cat_cols = [c for c in X.columns if c not in num_cols][:max_cat_cols]
    
    # 1. Continuous Boxplots
    if num_cols:
        fig, axes = plt.subplots(1, len(num_cols), figsize=(4 * len(num_cols), 4))
        if len(num_cols) == 1:
            axes = [axes]
        palette = ['#e74c3c', '#27ae60']
        for idx, col in enumerate(num_cols):
            sns.boxplot(data=df, x=target_col, y=col, hue=target_col, palette=palette, legend=False, ax=axes[idx], width=0.4)
            axes[idx].set_title(f"{col} by Target", fontweight='bold')
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
        plt.suptitle("Continuous Predictors vs Target Class Distribution", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
    # 2. Categorical Stacked Bar Plots
    if cat_cols:
        fig, axes = plt.subplots(1, len(cat_cols), figsize=(4.5 * len(cat_cols), 4))
        if len(cat_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(cat_cols):
            ct_norm = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            ct_norm.plot(kind='bar', stacked=True, color=['#e74c3c', '#27ae60'], ax=axes[idx], edgecolor='black', alpha=0.85)
            axes[idx].set_title(f"{col} Prevalence (%)", fontweight='bold')
            axes[idx].set_ylabel("Percentage (%)")
            axes[idx].legend(['Class 0', 'Class 1'], loc='upper right')
            axes[idx].grid(axis='y', linestyle='--', alpha=0.6)
        plt.suptitle("Categorical Predictors: Class Proportion Breakdown", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

# ==============================================================================
# HOW TO CALL THIS IN 1 LINE FOR ANY DATASET:
# ==============================================================================
# universal_classification_eda(df, target_col='pass')
```

---

## 🧠 Universal Interview Flashcard: Classification EDA

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Binary classification mein scatter plot kyu nahi banate?"** | Achha nahi lagta. | Binary target mein scatter plot sirf do horizontal lines (0 aur 1) dikhata hai. Yahan Grouped Boxplots, Overlaid KDE, aur Cross-Tabs use hote hain. |
| **"Categorical feature ka target se relation kaise test karte ho?"** | Correlation nikaalta hoon | Pearson correlation categorical data par nahi chalti! Cross-tabulation (`pd.crosstab`) aur Chi-Square ($\chi^2$) Test of Independence chalta hai. |
| **"Point-biserial correlation kya hota hai?"** | Do numbers ka correlation. | Ek continuous feature aur ek binary categorical feature (0/1) ke beech ka mathematically valid Pearson correlation Point-Biserial kehlata hai. |
| **"Chi-square test kyu lagate hain?"** | Normal distribution check karne ke liye. | Categorical feature aur Discrete target ke beech statistical association check karne ke liye ($p < 0.05$ matlab rishta statistically significant hai). |
