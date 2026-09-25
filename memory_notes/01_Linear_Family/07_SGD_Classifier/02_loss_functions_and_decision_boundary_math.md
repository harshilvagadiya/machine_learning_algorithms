# 📐 SGD Classifier Memory Notes: 02 - Loss Functions & Decision Boundary Math

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Classifier ka kaam do classes (0 aur 1) ke beech mein ek deewar (decision boundary $\mathbf{w}^T \mathbf{x} + b = 0$) khadi karna hota hai.  
> Lekin deewar kahan banegi aur kitna safety margin hoga, yeh **Loss Function** decide karta hai:  
> - **Hinge Loss (SVM):** Dono classes se barabar doori par sabse moti aur surakshit deewar banata hai (Maximum Margin).  
> - **Log-Loss (Logistic Regression):** Deewar ke sath-sath yeh bhi batata hai ki deewar se kitni door khade ho (0% se 100% Probability).  
> - **Modified Huber:** Agar kuch points galat side par feke gaye hain (noisy labels), toh unhe ignore karke deewar ko hilne nahi deta.

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📐 Section 1: The General Optimization Objective

$$\min_{\mathbf{w}} rac{1}{n} \sum_{i=1}^n L\left(y_i, \, \mathbf{w}^T \mathbf{x}_i + bight) + lpha R(\mathbf{w})$$

Jahan $y_i \in \{-1, +1\}$ binary class labels hote hain, aur $f(\mathbf{x}_i) = \mathbf{w}^T \mathbf{x}_i + b$ linear score hota hai.

Margin variable: $z_i = y_i \cdot (\mathbf{w}^T \mathbf{x}_i + b)$.
- Agar $z_i > 0$: Point sahi classify hua hai.
- Agar $z_i < 0$: Point galat classify hua hai.

---

## 🎯 Section 2: Mathematical Breakdown of the 4 Loss Functions

```
LOSS FUNCTION VISUAL COMPARISON (Margin z = y * f(x)):

      Loss
        ▲
      3 │          \ Perceptron: max(0, -z)
        │           \  
      2 │            \ Hinge (SVM): max(0, 1-z)
        │             \ 
      1 │              \  Log-Loss: log(1 + exp(-z))
        │                     0 └───────────────┬─┴──────────────────► Margin (z)
       -2              -1  0   1   2   3
                       Wrong │ Correct
```

### 1. `loss='hinge'` (Linear SVM - Default)
- Formula: $L(z) = \max(0, \, 1 - z)$
- **Behavior:** Agar point sahi side par hai aur margin se door hai ($z \ge 1$), toh loss = 0! Sirf margin violator points (Support Vectors) weights ko push karte hain.
- **Output:** Hard labels (`predict`). Probability estimate nahi milta.

### 2. `loss='log_loss'` (Logistic Regression)
- Formula: $L(z) = \log(1 + e^{-z})$
- **Behavior:** Smooth convex curve. Har ek point (chahe kitna bhi door ho) gradient mein halka sa hissa deta hai.
- **Output:** Calibrated probabilities `predict_proba()`. Business risk score ke liye ideal!

### 3. `loss='modified_huber'` (Outlier-Tolerant Probabilistic)
- Formula:
  $$L(z) = egin{cases} \max(0, 1 - z)^2 & 	ext{if } z \ge -1 \ -4z & 	ext{if } z < -1 \end{cases}$$
- **Senior Dev Advantage:** Yeh Hinge loss ki tarah margin banata hai, linear penalty deta hai extreme misclassifications par (noise resistant), aur **calibrated probabilities** bhi deta hai!

### 4. `loss='perceptron'` (Classical Perceptron)
- Formula: $L(z) = \max(0, -z)$
- Margin concept nahi hota; sirf misclassified points par linear penalty lagti hai.

---

## 🛡️ Section 3: Regularization Penalties (`penalty=...`)

- `'l2'`: Default Ridge penalty. Saare weights ko balance karta hai.
- `'l1'`: Lasso penalty. Text classification ya high-dimensional data mein **Sparsity** lata hai (faaltu words ke weights 0 kar deta hai).
- `'elasticnet'`: $L_1 + L_2$ mix (`l1_ratio` parameter ke sath). Correlated features ko group karke retain karta hai.

---

## ⚡ 10-Second Interview Flashcard: Loss Functions

| Loss Name | Mathematical Class | Supports `predict_proba()`? | Best Industry Use Case |
| :--- | :--- | :---: | :--- |
| `'hinge'` | Linear SVM | ❌ No | Maximum margin boundary, high-dimensional text / genomics |
| `'log_loss'` | Logistic Regression | ✅ **Yes** | Credit risk, churn prediction, healthcare diagnosis |
| `'modified_huber'` | Smooth Huber SVM | ✅ **Yes** | Noisy data, mislabeled training sets, fraud detection |
| `'perceptron'` | Perceptron | ❌ No | Fast baseline, linearly separable benchmark checks |
