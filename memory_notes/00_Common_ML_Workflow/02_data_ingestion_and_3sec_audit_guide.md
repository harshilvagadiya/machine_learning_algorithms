# 🌐 Common Linear Family: 02 - Data Ingestion & The 3-Second Health Audit

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Jab tum kisi doctor ke paas jaate ho, to kya doctor aate hi injection laga deta hai?  
> Nahi! Pehle pulse, BP aur body temperature check karta hai.  
> Machine Learning mein **Data Ingestion** ke baad seedha model fit karna gunah hai! Pehle 3 second ke andar data ka BP (Rows, Columns, Memory footprint aur Missingness) check karna zaroori hai!

---

## 🧭 The Ingestion Workflow

```
                        DATA INGESTION PIPELINE
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
ROBUST FILE INGESTION                                 3-SECOND HEALTH AUDIT
- Check file exists via Path                          - Total Rows & Columns
- Handle Encoding Traps (utf-8 -> latin-1)            - Memory Footprint (in MB)
- Auto-detect CSV vs. Excel                           - Dtypes overview & Quick Head/Tail
```

---

## ⚠️ Common Traps in Data Ingestion

1. **The FileNotFoundError Trap:**  
   Kabhi bhi hardcoded fragile path mat do. Hamesha Python ka standard `Path` module use karo.
2. **The `UnicodeDecodeError` Trap:**  
   Agar raw CSV file mein special characters hon (jaise Spanish/German text ya emojis), to standard `pd.read_csv()` crash ho jata hai. Hamesha fallback encoding `latin-1` try karo!
3. **The `ParserError: Buffer overflow caught` Trap:**  
   Agar CSV mein bade-bade multi-line text descriptions (jaise real estate descriptions) hon, to Pandas ka default C parser buffer overflow de deta hai. Iska ilaaj hai: **`engine='python'`** fallback!

---

## 📋 Copy-Paste Boilerplate: Robust Data Ingestion & 3-Sec Audit
*(Isko copy karke kisi bhi notebook ke Step 2 mein paste karo)*

```python
# ==============================================================================
# 🌐 STEP 2: ROBUST DATA INGESTION & 3-SECOND HEALTH AUDIT
# ==============================================================================
from pathlib import Path
import pandas as pd

def load_and_audit_dataset(file_path):
    """
    Loads dataset with automated encoding fallback and prints an enterprise 3-second audit.
    Loads dataset with automated encoding fallback and C-buffer overflow protection.
    Supports CSV and Excel files.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ File not found at: {path.resolve()}")
        
    # 1. Automated Ingestion with Encoding Fallback
    try:
        if path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print("⚠️ UTF-8 decoding failed. Retrying with 'latin-1' encoding...")
        df = pd.read_csv(path, encoding='latin-1')
    # Helper: Attempts default fast C engine, falls back to python engine on buffer overflow
    def _read_csv_safe(p, encoding):
        try:
            return pd.read_csv(p, encoding=encoding)
        except pd.errors.ParserError:
            print("⚠️ ParserError (Buffer Overflow). Retrying with engine='python'...")
            return pd.read_csv(p, encoding=encoding, engine='python')

    # 1. Automated Ingestion with Encoding & Engine Fallback
    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        try:
            df = _read_csv_safe(path, 'utf-8')
        except UnicodeDecodeError:
            print("⚠️ UTF-8 decoding failed. Retrying with 'latin-1' encoding...")
            df = _read_csv_safe(path, 'latin-1')

    # 2. Compute Health Metrics
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    n_numeric = df.select_dtypes(include=['number']).shape[1]
    n_categorical = df.select_dtypes(include=['object', 'category']).shape[1]
    total_nulls = df.isnull().sum().sum()

    # 3. Print 3-Second Audit Report
    print("=" * 70)
    print("📋 3-SECOND STRUCTURAL HEALTH AUDIT REPORT")
    print("=" * 70)
    print(f"File Source               : {path.name}")
    print(f"Total Observations (Rows) : {n_rows:,}")
    print(f"Total Features (Columns)  : {n_cols:,}")
    print(f"Numerical Columns         : {n_numeric} cols")
    print(f"Categorical/Text Columns  : {n_categorical} cols")
    print(f"Missing Value Cells       : {total_nulls:,} nulls ({total_nulls/(n_rows*n_cols)*100:.2f}%)")
    print(f"Memory Footprint in RAM   : {mem_mb:.2f} MB")
    print("=" * 70)
    
    return df

# Usage Example:
# df = load_and_audit_dataset("data/house_price/train.csv")
# display(df.head(3))
```

