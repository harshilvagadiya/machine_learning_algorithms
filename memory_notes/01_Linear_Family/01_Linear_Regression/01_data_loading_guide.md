# 📖 ML Memory Notes: Data Loading (Basic to Advanced)

> **Golden Rule of Data Loading:** 
> Agar data sahi load nahi hua, toh aage ka poora ML pipeline toot jayega. Data load karte hi pehle 3 second mein **Shape** aur **Head** check karna compulsory hai!

---

## 🟢 Level 1: Basic (Local Computer Se Load Karna)

### 1. Simple CSV Load
```python
import pandas as pd

# File padho aur table banao
df = pd.read_csv("data/my_file.csv")

# Hamesha yeh 2 lines run karo load hote hi:
print("Rows & Columns:", df.shape)
df.head()
```

### 2. Path Ka Lafda (FileNotFoundError Se Kaise Bachein?)
Jab Python bolta hai `FileNotFoundError: No such file or directory`:
- **Relative Path**: Jahan tumhari Jupyter Notebook baithi hai, wahan se rasta do:
  - Agar notebook `LinearRegression/` mein hai aur data uske andar `data/` mein hai:
    `"data/my_file.csv"`
- **Pathlib (Best Practice)**:
  ```python
  from pathlib import Path

  file_path = Path("data") / "my_file.csv"
  df = pd.read_csv(file_path)
  ```

---

## 🟡 Level 2: Intermediate (Kaggle Se Direct Download & Load)

> ⚠️ **The Kaggle Trap:** Kaggle website par likha `file_path` tumhare local computer ka path nahi hota. Uski jagah yeh **Bulletproof 3-Line Pattern** use karo jo kabhi fail nahi hota!

### The "Never-Fail" Kaggle Pattern:
```python
import os
import shutil
from pathlib import Path
import kagglehub
import pandas as pd

# Step 1: Sirf dataset ka naam do (Kaggle download karega)
cache_folder = kagglehub.dataset_download("yasserh/advertising-sales-dataset")

# Step 2: Apne project ke data folder mein copy karo
dest = Path("data/advertising_sales")
shutil.copytree(cache_folder, dest, dirs_exist_ok=True)

# Step 3: Python khud CSV file dhoondhega (Naam yaad rakhne ki tension khatam!)
csv_file = next(dest.glob("*.csv"))
df = pd.read_csv(csv_file)

print("Data Loaded! Shape:", df.shape)
df.head()
```

💡 **Kyu Best Hai?**
- File ka naam chahe kuch bhi ho (`Advertising.csv` ya `Advertising Budget and Sales.csv`), `next(dest.glob("*.csv"))` usko khud dhoondh leta hai!
- File offline tumhare project folder mein safely save ho jaati hai.

---

## 🔴 Level 3: Advanced (Real-World Industry Parameters)

Jab dataset ajeeb ho ya bohot bada ho, tab `pd.read_csv` ke andar yeh super-powers use hoti hain:

### 1. Faltu Index Column Hatana (`index_col`)
Kaggle datasets mein aksar ek faltu column hota hai: `Unnamed: 0` (jo bas 0, 1, 2... row number hota hai).
```python
# Unnamed: 0 ko alag feature banne se roko:
df = pd.read_csv("data.csv", index_col=0)
```

### 2. Comma (`,`) Ki Jagah Tab ya Semicolon Ho (`sep`)
Kabhi-kabhi data comma se nahi, semicolon (`;`) ya Tab space (`\t`) se alag hota hai:
```python
df = pd.read_csv("data.csv", sep=";")  # European style
df = pd.read_csv("data.tsv", sep="\t")  # Tab separated
```

### 3. Language / Special Characters Error (`encoding`)
Agar error aaye: `UnicodeDecodeError: 'utf-8' codec can't decode...`
```python
# Encoding badal do:
df = pd.read_csv("data.csv", encoding="latin1")  # Ya 'ISO-8859-1'
```

### 4. Huge Dataset (5 GB+ File) Se Sirf Kaam Ke Columns Padhna (`usecols`)
Agar dataset mein 100 columns hain lekin tumhe sirf 3 chahiye:
```python
df = pd.read_csv(
    "huge_data.csv", usecols=["Age", "Years of Experience", "Salary"]
)
# RAM bach gayi!
```

### 5. Sirf Chota Sample Padhna Testing Ke Liye (`nrows`)
Pehle sirf 500 rows load karke code test karo:
```python
df = pd.read_csv("huge_data.csv", nrows=500)
```

---

## 🟣 Level 4: Other Formats (CSV Ke Alawa Kya?)

| Format | Extension | Code | Kahan Use Hota Hai? |
| :--- | :--- | :--- | :--- |
| **Excel** | `.xlsx`, `.xls` | `pd.read_excel("data.xlsx")` | Finance, Business Reports |
| **JSON** | `.json` | `pd.read_json("data.json")` | Web APIs, Web Apps |
| **Parquet** | `.parquet` | `pd.read_parquet("data.parquet")` | Big Data, Cloud, Spark (Fastest!) |
| **SQL** | Database | `pd.read_sql("SELECT * FROM table", conn)` | Enterprise Production |

---

## ✅ Post-Load 3-Second Sanity Checklist

Data load hote hi apne aap se yeh 3 sawaal poocho:

```python
# 1. Kitni rows aur columns aayi?
print("Shape:", df.shape)

# 2. Pehli 5 rows theek dikh rahi hain?
display(df.head())

# 3. Columns ke types aur nulls kya hain?
df.info()
```

---

### 💡 Quick Memory Card:

```
Simple Local CSV:    pd.read_csv("data/file.csv")
Kaggle Auto-Find:    next(Path("data").glob("*.csv"))
Faltu Index Hatao:   pd.read_csv("...", index_col=0)
Badi File Ka Sample: pd.read_csv("...", nrows=1000)
Encoding Error Fix:  encoding="latin1"
```
