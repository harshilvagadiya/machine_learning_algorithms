# 🎛️ SGD Regressor Memory Notes: 03 - Learning Rate Schedules & Convergence

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Learning Rate ($\eta$) cycle ke pedal ki tarah hai:  
> - **Agar $\eta$ bohot bada rakha:** Cycle control se bahar ho jayegi aur gaddhe mein gir jayegi (Divergence / Loss $	o \infty$).  
> - **Agar $\eta$ bohot chhota rakha:** Choti tak pahunchne mein 10 saal lag jayenge (Slow convergence).  
> - **Learning Rate Schedule (Smart Strategy):** Shuruat mein jab gaddhe se door ho tab tez chalo, aur jaise-jaise gaddhe ke paas aao, cycle ki speed slow (brake) karte jao taaki minimum ke andar smoothly ruk sako!

---

## 🧭 Preprocessing Notice

> 📌 **COMMON STEPS NOTICE:** Data cleaning, splitting aur scaling ke liye dekhein:  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 🎛️ Section 1: The 4 Learning Rate Schedules in Scikit-Learn

Scikit-Learn ka `learning_rate` parameter decide karta hai ki step size $\eta(t)$ time $t$ ke sath kaise decay hoga:

```
SCHEDULE FORMULAS OVER TIME (t = step number):

1. 'constant':
   η(t) = η0  (Fixed step size, never changes)

2. 'invscaling' (Scikit-Learn Default):
   η(t) = η0 / (t^power_t)  (power_t default is 0.25)

3. 'optimal' (Léon Bottou's formula):
   η(t) = 1.0 / (alpha * (t0 + t))  (Automatically computed based on alpha)

4. 'adaptive' (Senior Developer's Favorite!):
   η(t) = η0 as long as training loss decreases.
   If loss stops improving for 'n_iter_no_change' epochs:
   η(t) = η(t) / 5.0  (Speed automatically reduced!)
```

---

## 🏆 Section 2: Why `learning_rate='adaptive'` is the Best in Practice

Jab aap `learning_rate='adaptive'` set karte ho:
1. Shuruat mein model full speed `eta0` par seekhta hai.
2. Agar validation loss plateau ho jata hai (improvement ruk jati hai), toh algorithm automatically learning rate ko $5	imes$ chhota kar deta hai.
3. Jab learning rate $10^{-6}$ se chhota ho jata hai, tab training automatically stop ho jati hai!
4. **Result:** Fast initial progress + Ultra-precise final convergence!

---

## 🛑 Section 3: Early Stopping & Stopping Criteria

Bina early stopping ke, model `max_iter=1000` tak andhadhundh chalta rahega chahe loss 50th epoch par hi converge ho chuki ho!

### Production Early Stopping Configuration:
```python
sgd = SGDRegressor(
    loss='squared_error',
    penalty='l2',
    alpha=0.001,
    learning_rate='adaptive',
    eta0=0.01,
    max_iter=5000,
    tol=1e-3,                # Minimum improvement threshold
    early_stopping=True,     # Automatically monitors internal validation split
    validation_fraction=0.1, # 10% data reserved for early stopping check
    n_iter_no_change=5,       # Agar 5 consecutive epochs tak loss tol se zyada nahi ghati, STOP!
    random_state=42
)
```

---

## ⚡ 10-Second Interview Flashcard: Learning Rates & Convergence

| Parameter | Recommended Value | What it Controls |
| :--- | :--- | :--- |
| `eta0` | `0.01` (or tuned `0.001` - `0.1`) | Initial learning rate / step size |
| `learning_rate` | `'adaptive'` or `'invscaling'` | Schedule policy for step decay |
| `early_stopping` | `True` | Prevents overfitting and saves compute hours |
| `n_iter_no_change` | `5` | Patience: kitne epochs tak wait kare bina improvement ke |
| `tol` | `1e-3` | Convergence tolerance threshold |
