# 🚀 SGD Regressor Memory Notes: 06 - Out-of-Core Learning & Streaming with `partial_fit`

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Socho aapke paas 50 GB ki CSV file hai, lekin aapke laptop mein sirf 8 GB RAM hai.  
> Agar aap `pd.read_csv('huge_file.csv')` karoge toh laptop blue screen / crash de dega!  
> Solution: **Out-of-Core Learning (`partial_fit`)**!  
> Hum file ko 10,000 - 10,000 rows ke chhote-chhote chunks mein padhenge, `partial_fit()` se model ko update karenge, aur chunk ko memory se fenk denge! Model 1 byte extra RAM liye bina 50 GB data seekh lega!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 💻 Section 1: The Out-of-Core Streaming Code Boilerplate

```python
# ==============================================================================
# 🚀 OUT-OF-CORE BIG DATA STREAMING WITH PARTIAL_FIT
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# 1. Initialize Online Scaler and Online SGD Model
scaler = StandardScaler()
sgd_streaming = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.0001,
    learning_rate='constant',  # partial_fit requires constant or invscaling
    eta0=0.01,
    random_state=42
)

csv_path = "huge_dataset.csv"
chunk_size = 10000  # 10,000 rows at a time (minimal RAM usage)

# 2. Pass 1: Fit Scaler incrementally on streaming chunks
for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
    # Perform basic cleaning
    chunk_clean = chunk.dropna()
    X_chunk = chunk_clean.drop(columns=['target'])
    scaler.partial_fit(X_chunk)

print("✅ Incremental feature statistics learned without loading full dataset into RAM!")

# 3. Pass 2: Train SGDRegressor incrementally across streaming chunks
n_epochs = 3
for epoch in range(n_epochs):
    chunk_count = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        chunk_clean = chunk.dropna()
        X_chunk = chunk_clean.drop(columns=['target'])
        y_chunk = chunk_clean['target']
        
        # Scale current chunk using our fitted streaming scaler
        X_chunk_scaled = scaler.transform(X_chunk)
        
        # Update model weights with this chunk
        sgd_streaming.partial_fit(X_chunk_scaled, y_chunk)
        chunk_count += 1
        
    print(f"🏁 Epoch {epoch + 1}/{n_epochs} complete ({chunk_count} chunks processed)!")

print("🎉 Streaming Big Data Training Completed with Zero Out-Of-Memory Risks!")
```

---

## ⚡ 10-Second Interview Flashcard: `fit()` vs. `partial_fit()`

| Feature | `model.fit()` | `model.partial_fit()` |
| :--- | :--- | :--- |
| **Data Requirement** | Entire dataset must be in memory at once | Takes 1 chunk/batch at a time |
| **Re-initialization** | Overwrites old weights from scratch | **Retains previous weights** and updates them |
| **Dataset Size Limit** | Limited by system RAM (e.g. 8-16 GB) | **Virtually Unlimited** (Terabytes of data) |
| **Supported Models** | Almost all Scikit-Learn models | Only iterative models (`SGDRegressor`, `SGDClassifier`, `MiniBatchKMeans`) |
