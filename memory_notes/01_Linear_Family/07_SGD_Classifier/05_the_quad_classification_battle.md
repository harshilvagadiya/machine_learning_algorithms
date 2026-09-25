# ⚔️ SGD Classifier Memory Notes: 05 - The Quad Classification Battle Showdown

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Classification Arena mein 4 linear maharathi aamne-saamne hain:  
> 1. **Logistic Regression (Batch / L-BFGS)**  
> 2. **Linear SVC (LibLinear Batch SVM)**  
> 3. **Perceptron (Classical Threshold)**  
> 4. **SGDClassifier (Stochastic Gradient Descent Champion)**  
> Dekhein ki kya fast iterative SGD batch models ke barabar ROC-AUC aur F1-Score deliver karta hai ya nahi!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## ⚔️ Step 9: The Quad Battle Implementation Code

```python
# ==============================================================================
# ⚔️ STEP 9: THE QUAD CLASSIFICATION BATTLE SHOWDOWN
# ==============================================================================
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import time

# 1. Initialize Contenders
models = {
    'Logistic Regression (L-BFGS)': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVC (LibLinear)':       LinearSVC(dual=False, random_state=42),
    'Perceptron':                   Perceptron(random_state=42),
    'SGD Classifier (Iterative)':   best_sgd
}

rows = []
for name, mod in models.items():
    t0 = time.time()
    mod.fit(X_train_final, y_tr)
    fit_time = (time.time() - t0) * 1000  # ms
    
    y_pred = mod.predict(X_test_final)
    
    # Check if model supports decision_function or predict_proba for ROC-AUC
    if hasattr(mod, "predict_proba"):
        y_prob = mod.predict_proba(X_test_final)[:, 1]
    elif hasattr(mod, "decision_function"):
        y_prob = mod.decision_function(X_test_final)
    else:
        y_prob = y_pred
        
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_prob)
    
    rows.append({
        'Model Name': name,
        'Train Time (ms)': f"{fit_time:.1f} ms",
        'Accuracy': f"{acc*100:.2f}%",
        'Precision': f"{prec*100:.2f}%",
        'Recall': f"{rec*100:.2f}%",
        'F1-Score': f"{f1*100:.2f}%",
        'ROC-AUC': f"{auc*100:.2f}%"
    })

battle_df = pd.DataFrame(rows)

print("=" * 95)
print("🏆 THE QUAD CLASSIFICATION BATTLE SHOWDOWN")
print("=" * 95)
print(battle_df.to_string(index=False))
print("=" * 95)
```

---

## 💡 Senior Developer Takeaways

1. **Accuracy is Identical, but SGD scales to 100x Data:** Batch Logistic Regression aur SGDClassifier small data par identical ROC-AUC deliver karte hain. Lekin 10 Lakh rows aate hi Batch models crash ho jaate hain jabki SGD microseconds mein execute hota hai.
2. **LinearSVC vs `SGDClassifier(loss='hinge')`:** Dono same loss optimize karte hain, lekin `SGDClassifier` out-of-core streaming (`partial_fit`) support karta hai jabki `LinearSVC` nahi karta!
