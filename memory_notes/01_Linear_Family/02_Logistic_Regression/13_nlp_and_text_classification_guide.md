# 🎯 Classification Memory Notes: 13 - Specialized Track: NLP & Text Classification Pipeline

> **The NLP Challenge:**
> Numbers ke tables (tabular data) mein columns pehle se defined hote hain (Age, Salary, Price).
> Lekin Text Data (Emails, SMS, Reviews) **Unstructured** hota hai! 
> Machine Learning model text nahi samajh sakta, isliye humein text ko ek **Numerical Matrix (Vectors)** mein convert karna padta hai!

---

## 🧭 The NLP Classification Pipeline

```
                    RAW TEXT EMAIL / SMS
                             │
                             ▼
                TEXT CLEANING & FEATURE EXTRACTION
                - Lowercasing (`text.lower()`)
                - Metadata features: Length, Uppercase words, Currency symbols ($ / £)
                - Stop words removal ("the", "is", "at")
                             │
                             ▼
                    TEXT VECTORIZATION
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
COUNT VECTORIZER (Bag of Words)       TF-IDF VECTORIZER (Industry Standard)
Sirf word frequency ginta hai.        Har word ko importance score deta hai.
Common words dominate karte hain!     Rare + Informative words ko boost karta hai!
                             │
                             ▼
                 SCIKIT-LEARN PIPELINE
              Vectorize -> Train Logistic Regression
                             │
                             ▼
                     PRODUCTION INFERENCE
              Raw String In  --->  Spam / Ham Out
```

---

## 🟢 Level 1: Engineered Text Metadata Features

Raw text ko NLP vectorizer mein daalne se pehle, kuch direct numerical features extract karna bahut powerful hota hai (Spam emails ki khas nishaaniyan):

```python
import pandas as pd

# 1. Total character length
df['char_count'] = df['message'].apply(len)

# 2. Total word count
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# 3. Capital letter count (Spam mein loud shouting hoti hai: "FREE!! WINNER!!")
df['caps_count'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()))

# 4. Currency and exclamation marks count
df['exclamation_count'] = df['message'].apply(lambda x: x.count('!'))
df['currency_count'] = df['message'].apply(lambda x: x.count('$') + x.count('£'))

# 5. Presence of numbers/digits
df['has_digits'] = df['message'].apply(lambda x: int(any(c.isdigit() for c in x)))
```

---

## 🟡 Level 2: Bag of Words vs. TF-IDF

### 1. Bag of Words (`CountVectorizer`)
- Pure corpus ki ek dictionary banata hai.
- Har email mein kaunsa word kitni baar aaya, bas uska count matrix banata hai.
- **Problem:** Jo words harmless aur common hain ("please", "today", "call") woh bada count le aate hain, aur jo actually spam keywords hain ("jackpot", "urgent") woh dab jaate hain.

---

### 2. TF-IDF Vectorizer (Term Frequency - Inverse Document Frequency)
Yeh har word ko ek mathematical weight deta hai:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

#### A. Term Frequency (TF):
Word $t$ us specific email $d$ mein kitni baar aaya:
$$\text{TF}(t, d) = \frac{\text{Count of word } t \text{ in email } d}{\text{Total words in email } d}$$

#### B. Inverse Document Frequency (IDF):
Word $t$ poori duniya (saari emails) mein kitna rare ya aam hai:
$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
- $N$ = Total emails.
- $\text{DF}(t)$ = Kitni emails mein word $t$ maujood hai.

#### 💡 The Intuition:
| Word Type | TF in this email | DF across all emails | IDF Weight | Final TF-IDF Score |
| :--- | :--- | :--- | :--- | :--- |
| **"the", "is", "and"** | High (5 times) | Extremely High (99% emails) | Near ZERO | **0.0 (Suppressed!)** |
| **"lottery", "claim"** | High (3 times) | Rare (only 2% spam emails) | Very HIGH | **HIGH (Spam Signal!)** |

---

## 🔴 Level 3: Mastering `TfidfVectorizer` Parameters

Scikit-Learn ka `TfidfVectorizer` default parameters par theek hota hai, lekin production mein yeh parameters tuning ke liye golden hain:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    lowercase=True,             # Sabhi text ko lowercase karega
    stop_words='english',       # 'the', 'is', 'in' jaise 318 useless words hata dega
    max_features=3000,          # Top 3000 most important words hi rakhega (memory bachti hai)
    min_df=2,                   # Jo word 2 se kam emails mein aaya ho use drop karo (typos filter)
    max_df=0.95,                # Jo word 95% se zyaada emails mein ho use drop karo
    ngram_range=(1, 2),         # Unigrams ("claim") aur Bigrams ("claim now", "free prize") dono seekhega!
    sublinear_tf=True           # TF scaling ko log(TF) karega taaki 10 baar repeat karne par spammer cheat na kare
)
```

---

## 🛠️ Level 4: The Clean Scikit-Learn Pipeline (Best Practice)

Production mein data preprocessing aur model training ko alag-alag rakhne ke bajaye **Pipeline** mein baandh do. Isse data leakage 0% ho jaati hai!

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1. End-to-End Pipeline define karo
spam_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))),
    ('classifier', LogisticRegression(C=1.0, solver='liblinear', random_state=42))
])

# 2. Train par Fit karo (Automatic Vectorization + Training!)
spam_pipeline.fit(X_train_text, y_train)

# 3. Test par direct predict karo (No manual transform needed!)
y_pred = spam_pipeline.predict(X_test_text)
y_prob = spam_pipeline.predict_proba(X_test_text)[:, 1]
```

---

## 🚀 Level 5: Live Unseen Text Prediction Simulator

Model train hone ke baad live real-world inputs par test karne ka template:

```python
def predict_email(text_message, pipeline, threshold=0.5):
    prob_spam = pipeline.predict_proba([text_message])[0, 1]
    is_spam = int(prob_spam >= threshold)
    
    result = {
        'Input Text': text_message,
        'Spam Probability': f"{prob_spam * 100:.2f}%",
        'Prediction': "🚨 SPAM" if is_spam == 1 else "✅ HAM (Legitimate)",
        'Confidence': f"{max(prob_spam, 1 - prob_spam) * 100:.1f}%"
    }
    return result

# Test samples:
print(predict_email("Hey mate, are we still meeting for coffee at 4pm today?", spam_pipeline))
print(predict_email("CONGRATULATIONS! You won £1000 cash. Reply CLAIM to claim your prize now!", spam_pipeline))
```

---

## 🧠 Quick Revision Checklist: NLP Text Classification
- [ ] Raw text ko inspect karke cleaning zaroorat dekhi (HTML tags, URLs, encoding)?
- [ ] Engineered features (length, caps, symbols) check kiye?
- [ ] Bag of Words ke bajaye TF-IDF vectorizer use kiya?
- [ ] `stop_words='english'` aur `ngram_range=(1, 2)` lagaya?
- [ ] Vectorizer aur Classifier ko `Pipeline` ke andar pack kiya?
- [ ] Live inference function se custom unseen emails test kiye?

