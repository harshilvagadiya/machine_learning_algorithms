# ✂️ Lasso Regression Memory Notes: 03 - Geometric Intuition: Diamond vs. Circle

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> *"Lasso mein weights exact zero kyu ban jaate hain, jabki Ridge mein hamesha 0.0001 jaisa chhota number bacha rehta hai?"*  
> Iska sabse khoobsurat jawab Geometry (Shapes) mein chhupa hai:  
> - **Ridge ki constraint ek GOL BOWL (Circle) hoti hai.**  
> - **Lasso ki constraint ek TEEKHA HEERA (Diamond) hoti hai!**  
> Heere (Diamond) ke chaar kone (corners) theek axes par hote hain jahan ek weight exact **0** hota hai!

---

## 🧭 Preprocessing Notice:
> 📌 *Data Ingestion, Cleaning, Scaling, aur Train-Test Split ke standard steps ke liye refer karein:*  
> ➡️ **[00_Common_Linear_Family Documentation](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📐 Section 1: Constrained Optimization Formulation

Dono models ko hum is tarah dekh sakte hain:  
*"Error (RSS) ko kam se kam karo, lekin tumhare paas weight ka ek budget ($t$) hai!"*

### 1. Ridge ($L_2$) Constraint:
$$w_1^2 + w_2^2 \le t \quad \implies \text{Equation of a CIRCLE (Gol Katora)}$$

### 2. Lasso ($L_1$) Constraint:
$$|w_1| + |w_2| \le t \quad \implies \text{Equation of a DIAMOND (Teekha Heera)}$$

---

## 🎨 Section 2: Visualizing the Collision (Contours vs. Constraints)

```
        RIDGE (CIRCLE)                               LASSO (DIAMOND)

              w2                                           w2
              ▲                                            ▲
              │                                            │  (0, t)
          ┌───┼───┐                                      / │ \
       /      │      \                                  /  │  \
      │       │       │                                /   │   \
──────┼───────┼───────┼──────► w1            ──────(-t,0)──┼───(t,0)──► w1
      │       │       │                                \   │   /
       \      │      /                                  \  │  /
          └───┼───┘                                      \ │ /
              │                                            ▼  (0, -t)
              ▼

Collision Point: Smooth Arc                 Collision Point: Sharp Corner on Axis!
w1 = 0.35, w2 = 0.42                        w1 = 0.0, w2 = 0.78
(Neither weight is zero!)                   (w1 is EXACTLY ZERO!)
```

---

## 🔍 Section 3: Kyu Ellipse Hamesha Heere ke Kone par Takrata hai?

1. **Error Contours (Ellipses):**  
   OLS solution center mein hota hai. Jaise-jaise error badhta hai, concentric andakaar (elliptical) rings bahar ki taraf failti hain.
2. **Pehle Kahan Takrayega?**  
   - **Circle (Ridge):** Har taraf se gol aur smooth hai. Ellipse circle ke kisi curved surface par tangent banata hai. Wahan $w_1 \neq 0$ aur $w_2 \neq 0$.
   - **Diamond (Lasso):** Iske chaar teekhe corners hote hain jo theek axes par hote hain: $(t, 0), (-t, 0), (0, t), (0, -t)$.  
   Kyunki corners bahar ki taraf nikle hote hain, failti hui ellipse **lagbhag hamesha kisi kone (corner) par pehle touch karti hai!**
3. **Corner ka Matlab:**  
   Jab ellipse axis ke corner par touch karegi:
   $$\text{Touch at } (0, t) \implies w_1 = 0.0 \quad \text{aur} \quad w_2 = t$$
   Yeh feature $w_1$ ko **exact ZERO** bana deta hai!

---

## ⚡ 10-Second Interview Flashcard

| Question | Junior Developer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"Lasso feature selection kyu karta hai?"** | Kyunki formula mein absolute value hai. | $L_1$ norm ki constraint space ek **polytope / diamond** hoti hai jiske **vertices (corners) axes par** hote hain. Error contours aksar in sharp vertices par touch karte hain, jisse non-informative features ke weights exact $0.0$ ho jaate hain. |

