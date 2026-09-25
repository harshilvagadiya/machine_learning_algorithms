# 🎯 SGD Classifier Memory Notes: 03 - Classification Metrics & Decision Threshold Tuning

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Regression mein hum $R^2$ aur RMSE dekhte the.  
> Classification mein **Accuracy ek bohot bada Dhokha (Trap) hai!**  
> Socho 10,000 credit card transactions mein se sirf 10 fraud hain:  
> Agar aapka model ek aalsi chaprasi hai jo har transaction ko bol deta hai *"No Fraud"*, toh uski Accuracy $99.9\%$ hogi! Lekin bank ka 1 Crore ka nuksan ho jayega!  
> Isliye Senior Developers **Precision, Recall, F1-Score, ROC-AUC, aur Decision Threshold Tuning** use karte hain!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 🧩 Section 1: The Confusion Matrix Foundation

```
                               ACTUAL TRUTH (Reality)
                               Positive (1)        Negative (0)
                       ┌──────────────────────┬──────────────────────┐
       Predicted (1)   │ True Positive (TP)   │ False Positive (FP)  │
PREDICTION             │ "Chor pakad liya"    │ "Bekasoor par Fine"  │
                       ├──────────────────────┼──────────────────────┤
       Predicted (0)   │ False Negative (FN)  │ True Negative (TN)   │
                       │ 🚨 "Chor bhaag gaya!"│ "Normal Customer"    │
                       └──────────────────────┴──────────────────────┘
```

---

## ⚖️ Section 2: Precision vs. Recall (The Ultimate Business Trade-off)

### 1. Precision (Quality / Trust):
$$	ext{Precision} = rac{	ext{TP}}{	ext{TP} + 	ext{FP}}$$
- *"Jab mere model ne kaha Fraud hai, toh kitni baar wo sach nikla?"*
- **High Precision Chahiye:** Spam Filter (Important email spam folder mein nahi jaana chahiye).

### 2. Recall (Sensitivity / Coverage):
$$	ext{Recall} = rac{	ext{TP}}{	ext{TP} + 	ext{FN}}$$
- *"Duniya mein jitne chor the, unme se kitne percent chor pakde gaye?"*
- **High Recall Chahiye:** Cancer Detection ya Credit Card Fraud (Mareez chhoothna nahi chahiye!).

### 3. F1-Score (The Harmonic Balance):
$$	ext{F}_1 = 2 \cdot rac{	ext{Precision} \cdot 	ext{Recall}}{	ext{Precision} + 	ext{Recall}}$$

---

## 🎚️ Section 3: Decision Threshold Tuning (The 0.5 Trap)

Default `predict()` hamesha probability threshold **0.50 (50%)** par cut karta hai:

```python
# Standard Default (May Miss Frauds)
y_pred_default = (y_prob >= 0.50).astype(int)

# Senior Developer Custom Threshold for High Recall (Catches More Frauds!)
# Agar 30% shak bhi hai, toh review mein daal do!
custom_threshold = 0.30
y_pred_tuned = (y_prob >= custom_threshold).astype(int)
```

---

## ⚡ 10-Second Interview Flashcard: Classification Evaluation

| Metric | Formula | When to Optimize? |
| :--- | :--- | :--- |
| **ROC-AUC** | Area Under True Positive vs False Positive Rate | Overall model discrimination power across ALL thresholds |
| **Precision** | $	ext{TP} / (	ext{TP} + 	ext{FP})$ | When False Positives are expensive (Spam, Arrests, Account bans) |
| **Recall** | $	ext{TP} / (	ext{TP} + 	ext{FN})$ | When False Negatives are dangerous (Cancer, Fraud, Defects) |
| **PR-AUC** | Area Under Precision-Recall Curve | Best metric for severely imbalanced datasets ($< 5\%$ positives) |
