# 📖 ML Memory Notes: Model Training, Evaluation & Diagnostics (Basic to Advanced)

> **Golden Law of Modeling:** 
> **"Fit on Train, Transform on Test!"**
> Test data ko model training ke dauraan chhoona bhi paap (Data Leakage) hai! Test set sirf aur sirf aakhri imtihaan (Final Exam) ke liye hota hai!

---

## 🧭 The End-to-End Regression Roadmap

```
             CLEAN NUMERIC DATA (X, y)
                         │
                         ▼
             TRAIN-TEST SPLIT (80% / 20%)
             `train_test_split(..., random_state=42)`
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
   TRAIN SET (80%)                TEST SET (20%)
   (Sirf ispe seekhega)           (Parde ke peeche band)
          │                             │
          ▼                             ▼
   FIT SCALER / MODEL             TRANSFORM & PREDICT
   `model.fit(X_train, y_train)`  `y_pred = model.predict(X_test)`
                         │
                         ▼
             EVALUATION METRICS & DIAGNOSTICS
             (R², MAE, RMSE, Residual Plots)
                         │
                         ▼
             SERIALIZE & PERSIST MODEL
             `joblib.dump(model, 'model.pkl')`
```

---

## 🟢 Level 1: Train-Test Split (The Imtihaan Rule)

Kyun karte hain Train-Test Split?
- Agar tum student ko wahi questions paper doge jo usne ghar pe practice kiye the, toh woh 100/100 le aayega (Rote learning / Overfitting).
- Asali dimaag tab pata chalta hai jab uske samne **naye unseen questions** aate hain!

```python
from sklearn.model_selection import train_test_split

# 80% Data padhai ke liye (Train), 20% Final Exam ke liye (Test)
# random_state=42 lagane se har baar same split aayega (Reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🏋️ Training Set: {X_train.shape[0]} rows (80%)")
print(f"🧪 Testing Set:  {X_test.shape[0]} rows (20%)")
```

---

## 🟡 Level 2: Feature Scaling (Standardization vs. Normalization)

Jab features ke scales alag-alag hon:
- `House Size`: 500 se 5,000 sqft
- `Bathrooms`: 1 se 5
- `Price`: $50,000 se $1,000,000

| Scaler | Formula | Range | Kab Use Karein? |
| :--- | :--- | :---: | :--- |
| **`StandardScaler`** | $z = \frac{x - \mu}{\sigma}$ | Mean = 0, Std = 1 | **Default choice!** Bell curve data aur Ridge/Lasso ke liye best. |
| **`MinMaxScaler`** | $x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$ | $[0, 1]$ | Jab data strictly bounded chahiye ho (Image pixels, Neural Networks). |
| **`RobustScaler`** | $x_{rob} = \frac{x - \text{median}}{\text{IQR}}$ | Median = 0 | Jab data mein **bohot zyada Outliers** hon jinhe drop nahi karna. |

### ⚠️ The Golden Leakage-Free Rule:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Rule: Train par FIT_TRANSFORM karo, Test par SIRF TRANSFORM karo!
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 🔴 Level 3: The 3 Core Regression Algorithms

### 1. Ordinary Least Squares (OLS) / `LinearRegression()`
- **Kaise Kaam Karta Hai?** Seedhi line khinch kar squared errors ($\sum (y - \hat{y})^2$) ko minimize karta hai.
- **Pros:** Super fast, simple, mathematical, coefficients directly explainable hote hain.
- **Cons:** Outliers aur Multicollinearity se bohot jaldi disturb ho jata hai.

```python
from sklearn.linear_model import LinearRegression

ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
y_pred = ols_model.predict(X_test)
```

---

### 2. Ridge Regression ($L_2$ Regularization)
- **Kaise Kaam Karta Hai?** OLS ke loss function mein coefficients ke square ka penalty jod deta hai:
  $$\text{Loss} = \text{RSS} + \alpha \sum \beta_j^2$$
- **Kya Karta Hai?** Coefficients ko chhota (shrink) kar deta hai, lekin kisi ko **zero nahi karta**!
- **Kab Use Karein?** Jab features aapas mein correlated hon (Multicollinearity) ya model overfit ho raha ho.

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
```

---

### 3. Lasso Regression ($L_1$ Regularization)
- **Kaise Kaam Karta Hai?** Loss function mein absolute values ka penalty jodta hai:
  $$\text{Loss} = \text{RSS} + \alpha \sum |\beta_j|$$
- **Kya Karta Hai?** Faltu features ke coefficients ko **EXACTLY ZERO (0.0)** bana deta hai!
- **Superpower:** **Automated Feature Selection!** Agar 100 features hain, toh Lasso bekaar 80 features ko uda kar 0 kar dega aur sirf top 20 ko rakhega!

```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)

# Zero features check karo:
zero_features = X.columns[lasso_model.coef_ == 0].tolist()
print("Lasso eliminated these features:", zero_features)
```

---

## 📊 Level 4: Evaluation Metrics (Report Card)

Regression mein model kitna accha hai, yeh 3 numbers batate hain:

| Metric | Formula | Real-Life Matlab | Kab Use Karein? |
| :--- | :---: | :--- | :--- |
| **MAE** *(Mean Absolute Error)* | $\frac{1}{n}\sum \|y - \hat{y}\|$ | Average galti natural units mein (e.g. ₹11,300 error). | **Business clients ko samjhane ke liye sabse best!** Outliers se influence nahi hota. |
| **RMSE** *(Root Mean Squared Error)* | $\sqrt{\frac{1}{n}\sum (y - \hat{y})^2}$ | Badi galtiyon ko heavy penalty deta hai (Square karke root lena). | Jab badi galtiyaan (big blunders) company ke liye dangerous hon. |
| **$R^2$ Score** *(R-Squared)* | $1 - \frac{SS_{res}}{SS_{tot}}$ | **0 se 1 (0% se 100%)**: Kitne percent variation model ne explain kiya. | Model ki overall predictive capability batane ke liye. |

```python
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.4f} (~{r2*100:.1f}%)")
print(f"MAE:      {mae:,.2f}")
print(f"RMSE:     {rmse:,.2f}")
```

> ⚠️ **The MSE Trap:** Kabhi bhi non-technical boss ya business team ko raw **MSE (Mean Squared Error)** report mat karna! Kyunki agar currency Rupees hai, toh MSE "Rupees Squared" ($₹^2$) ho jata hai (jaise ₹760 million squared), jiska koi practical meaning nahi hota! Hamesha **MAE** ya **RMSE** report karo!

---

## 🩺 Level 5: Model Diagnostics (Checking Model Health)

Sirf metrics dekh kar khush mat ho — model ke andar ka health check (Residuals) karo!
Residual matlab: **$e = y_{\text{actual}} - \hat{y}_{\text{pred}}$** (Asali value aur model ke andaze ka farq).

### Diagnostic 1: Actual vs. Predicted Plot
Dots jitne red 45° line ($y = x$) ke paas honge, model utna super accurate hai!

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color="#1f77b4", alpha=0.5, edgecolors="k")
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
plt.title("Actual vs. Predicted Plot", fontweight="bold")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
```

### Diagnostic 2: Residual Plot (Homoscedasticity Check)
Errors zero line ke aas-paas **randomly bikhre** hone chahiye. Agar koi funnel shape ya pattern dikhe, matlab model variance unstable hai!

```python
residuals = y_test - y_pred

plt.figure(figsize=(7, 4))
plt.scatter(y_pred, residuals, color="#ff7f0e", alpha=0.5, edgecolors="k")
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.title("Residual Plot (Error vs Predicted)", fontweight="bold")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.show()
```

---

## 💾 Level 6: Model Persistence & Production Deployment

Model train hone ke baad usko hard drive par save karna zaroori hai taaki website, app, ya FastAPI backend mein load karke live predictions li ja sakein:

```python
import joblib
from pathlib import Path

# 1. Model aur Feature columns dono ko ek dictionary mein bundle karo:
export_package = {
    "model": model,
    "feature_columns": list(X.columns)
}

# 2. Disk par save karo (.pkl format)
model_path = Path("models") / "final_regression_model.pkl"
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(export_package, model_path)
print("✅ Model successfully saved!")

# 3. Reload & Live Smoke Test
loaded_pkg = joblib.load(model_path)
live_model = loaded_pkg["model"]

# Single new row prediction
sample_pred = live_model.predict(X_test.iloc[[0]])
print(f"🎯 Live Prediction: {sample_pred[0]:,.2f}")
```

---

## 🚫 The 4 Deadly Sins of Modeling (Interview Traps)

| ❌ Deadly Sin | Asali Nuqsan | Sahi Tarika ✅ |
| :--- | :--- | :--- |
| **`scaler.fit_transform(X)` poore data par lagana** | **Data Leakage!** Test set ka mean aur variance train mein leak ho jata hai. | Hamesha pehle Train-Test Split karo, fir `scaler.fit(X_train)` aur `scaler.transform(X_test)`! |
| **Regularization bina Scaler ke chalana** | Ridge/Lasso badi values wale feature (e.g. Size=2000) ko bina wajah zyada penalize kar denge! | Ridge aur Lasso ke pehle **hamesha `StandardScaler` lagao**! |
| **Skewed Target par seedha OLS lagana** | Extreme high values poori regression line ko kheench kar barbaad kar deti hain. | Target skewness > 1.0 ho toh **`np.log1p(y)`** par train karo aur aakhir mein **`np.expm1()`** se wapas natural unit mein laao! |
| **Sirf $R^2$ dekh kar khush ho jana** | Agar dataset mein 10 extreme outliers hain, toh $R^2$ fake high dikha sakta hai jabki MAE bekaar ho. | Hamesha **$R^2$ ke sath MAE aur Residual Plot** dono inspect karo! |

