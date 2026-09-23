# 📖 ML Memory Notes: D - Duplicate Rows Check (Basic to Advanced)

> **Golden Rule of Duplicates:** 
> Hamesha check karo: **Duplicate kyu aaya?** 
> Agar database bug ya accidental repeat entry hai toh **DROP** karo. Lekin agar natural data distribution hai (jaise same age, same experience, same salary) toh **KEEP** karo!

---

## 🧭 The Duplicate Decision Tree

```
                          DUPLICATE MILA?
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
   ID / KEY COLUMN REPEAT HAI?                   GENERIC ATTRIBUTES REPEAT HAIN?
   (Jaise same Student_ID ya                     (Jaise 2 alag log hain, dono
   Email do baar aa gaya)                        ki age 28 aur degree Master's hai)
        │                                               │
        ▼                                               ▼
   🔥 100% DROP KARO!                            🤔 CONTEXT CHECK KARO!
   `df.drop_duplicates()`                         - Salary/Census survey hai: KEEP
                                                  - Web scraper duplicate hai: DROP
```

---

## 🟢 Level 1: Counting Duplicates (The `.count()` vs `.sum()` Trap)

> ⚠️ **Common Trap:** 
> `df.duplicated().count()` likhoge toh **Total Rows (e.g. 6704)** de dega! 
> Kyunki `.count()` True aur False dono ko ginta hai. **Hamesha `.sum()` use karo!**

```python
# 1. Sahi Tareeqa:
duplicate_count = df.duplicated().sum()
duplicate_pct = (duplicate_count / len(df)) * 100

print(f"Total Duplicate Rows: {duplicate_count} ({duplicate_pct:.2f}%)")
```

---

## 🟡 Level 2: Duplicate Rows Ko Inspect Karna (`keep=False`)

Sirf count dekh kar andha decision mat lo. Original aur duplicate ko aamne-saamne rakh kar dekho:

```python
# `keep=False` ka matlab: Original aur Duplicate dono ko ek saath dikhao!
duplicate_pairs = df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:2]))

print(f"Total Rows Involved in Duplication: {len(duplicate_pairs)}")
print("Sample Duplicate Pairs (Aamne-Saamne):")
display(duplicate_pairs.head(6))
```

💡 **Fayda:** Isse tumhe pata chal jata hai ki sach mein poori row identical hai ya bas kuch columns.

---

## 🔴 Level 3: Dropping Duplicates (The 3 Options in `drop_duplicates`)

Jab drop karna ho, toh Pandas 3 options deta hai:

```python
# Option 1: Pehli wali rakho, baaki duplicate udao (MOST COMMON - DEFAULT)
df_clean = df.drop_duplicates(keep="first").copy()

# Option 2: Aakhri wali rakho, purani wali udao
df_clean = df.drop_duplicates(keep="last").copy()

# Option 3: Sabhi duplicate ko uda do (Sirf unique single rows bachengi)
df_clean = df.drop_duplicates(keep=False).copy()
```

### Pro Level: Kisi Specific Column Par Duplicate Check Karna (`subset`)
Agar tum chahte ho ki agar **Email** ya **User_ID** match ho jaye, toh duplicate maano:

```python
# Sirf 'Email' column ke basis par duplicate hatao:
df_clean = df.drop_duplicates(subset=["Email"], keep="first").copy()
```

---

## 🚨 Level 4: The Dangerous ML Trap — "Data Leakage by Duplicates"

Machine Learning mein duplicate data ka sabse bada khatra kya hota hai?

Imagine karo:
- Row 42 aur Row 150 ek hi bande ka duplicate data hain.
- Train-Test Split hua:
  - **Row 42 chali gayi `X_train` mein!**
  - **Row 150 chali gayi `X_test` mein!**

Model ne train data mein Row 42 ko ratta maar liya. Ab jab test data mein wahi exact sawal aaya (Row 150), toh model ne 100% correct answer de diya!
Tum sochoge: *"Mera model toh genius hai!"* 
Lekin asliyat mein **exam se pehle paper leak ho gaya tha!**

> 💡 **Golden ML Defense:** 
> Agar data mein identical duplicate rows hain jo accidental hain, toh **TRAIN-TEST SPLIT SE PEHLE UNHEIN DROP KARO**!

---

## 🟣 Level 5: Hidden Duplicates (Spaces Aur Capital Letters Ka Jhol!)

Kabhi-kabhi rows duplicate hoti hain, lekin Pandas unhe duplicate nahi bolta kyunki:
- Ek jagah likha hai: `"Google"`
- Doosri jagah likha hai: `"Google "` (peeche space laga hai!)
- Teesri jagah likha hai: `"google"` (chota 'g' hai!)

### 💡 Unmasking Hidden Duplicates:
```python
# Text columns se extra space hatao aur lowercase karo:
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Ab check karo: naye duplicates pakde jayenge!
print("Clean hone ke baad duplicates:", df.duplicated().sum())
```

---

## 📊 Visualizing Duplicate Data

```python
import matplotlib.pyplot as plt

total_rows = len(df)
dups = df.duplicated().sum()
uniques = total_rows - dups

plt.figure(figsize=(6, 4))
plt.pie([uniques, dups], 
        labels=["Unique Records", "Duplicate Rows"],
        colors=["#2ca02c", "#d62728"], 
        autopct="%1.1f%%", 
        startangle=90,
        explode=(0, 0.1 if dups > 0 else 0))
plt.title(f"Unique vs. Duplicate Records (Total: {total_rows:,})", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()
```

---

## 📋 The "Copy-Paste" Template for Any Project

```python
# --- STEP 2.3: D - DUPLICATE ROWS AUDIT ---

dup_count = df.duplicated().sum()
dup_pct = (dup_count / len(df)) * 100

if dup_count > 0:
    print(f"⚠️ Alert: {dup_count} duplicate rows found ({dup_pct:.1f}% of total data).")
    # Agar drop karna ho:
    # df = df.drop_duplicates(keep="first").copy()
    # print(f"Duplicates removed! New shape: {df.shape}")
else:
    print("✅ Perfect! Zero duplicate rows detected.")
```
