# 🚀 SGD Classifier Memory Notes: 06 - Out-of-Core Streaming Classification (`partial_fit`)

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho bank ke paas 1 Crore live credit transactions hain ya Twitter/X par har second 10,000 tweets aa rahe hain jinka sentiment classify karna hai.  
> Poora data ek saath kabhi computer ki RAM mein nahi aa sakta!  
> Solution: **`model.partial_fit()`**!  
> Har incoming data batch ko stream karke model ko online update karo!  
> 🚨 **Critical Classification Rule:** Pehle `partial_fit()` call mein **`classes=np.unique(all_classes)`** batana zaroori hai taaki model ko pata ho kitni total classes hain!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 💻 Section 1: The Out-of-Core Streaming Pattern Boilerplate

```python
# ==============================================================================
# 🚀 OUT-OF-CORE STREAMING CLASSIFICATION WITH PARTIAL_FIT
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# 1. Initialize Streaming Classifier
sgd_streaming = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.0001,
    learning_rate='constant',  # partial_fit requires constant or invscaling
    eta0=0.01,
    random_state=42
)

csv_path = "huge_transactions.csv"
chunk_size = 50000  # 50,000 transactions at a time
all_classes = np.array([0, 1])  # 0: Legitimate, 1: Fraud

# 2. Online Streaming Training Loop
chunk_idx = 0
for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    chunk_clean = chunk.dropna()
    X_chunk = chunk_clean.drop(columns=['is_fraud'])
    y_chunk = chunk_clean['is_fraud']
    
    # CRITICAL: Pass 'classes' argument on the very first batch!
    if chunk_idx == 0:
        sgd_streaming.partial_fit(X_chunk, y_chunk, classes=all_classes)
    else:
        sgd_streaming.partial_fit(X_chunk, y_chunk)
        
    chunk_idx += 1
    if chunk_idx % 5 == 0:
        print(f"📦 Processed {chunk_idx * chunk_size:,} transactions online with 0 extra RAM!")

print("🎉 Out-of-Core Streaming Classifier successfully trained on massive data!")
```

---

## ⚡ 10-Second Interview Flashcard: Classification `partial_fit()`

| Rule | Why Mandatory? |
| :--- | :--- |
| **`classes=np.unique(...)` Argument** | Classification mein model ko shuruat mein batana padta hai ki total classes kitni hain (e.g. 0 aur 1, ya multiclass 0, 1, 2). Agar pehle batch mein sirf Class 0 aayi aur argument nahi diya, toh model crash ho jayega! |
| **Learning Rate Policy** | `partial_fit` ke sath `'adaptive'` kaam nahi karta kyunki epochs track nahi hote. Isliye `'constant'` ya `'invscaling'` use kiya jata hai. |
