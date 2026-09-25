# ⚡ Algorithm 07: SGD Classifier (Large-Scale & Out-of-Core Classification) — Complete Memory Notes Suite

> **Dumb Student Master Summary (Ek Line Mein):**  
> `SGDClassifier` Scikit-Learn ka sabse tez linear classification engine hai jo ek single parameter (`loss`) badalne se **Logistic Regression (`log_loss`)**, **Linear SVM (`hinge`)**, ya **Perceptron** ban jata hai, aur **1 Crore rows** par bina kisi memory crash ke sub-millisecond inference deliver karta hai!

---

## 🧭 Preprocessing & Universal Pipeline Protocol

> 📌 **CRITICAL RULE (DO NOT REPEAT COMMON STEPS):**  
> Linear Family ke pehle 7 steps (Libraries, Ingestion, Sanitization, Feature Segregation, Missing Values Hygiene, Zero-Leakage Split with Target Guard, Class Imbalance Profiler, aur Master ColumnTransformer) har project mein **identical** hote hain!  
> Unke detailed concept notes, mathematical reasoning aur production boilerplate code ke liye refer karein:  
> ➡️ **[00_Common_Linear_Family Master Suite](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/00_Common_Linear_Family/README.md)**

---

## 📚 Step-by-Step SGD Classifier Memory Notes Suite

Humne SGD Classifier ke core concepts ko 7 dedicated guides mein organize kiya hai:

| Guide # | Document Title | What You Will Learn (In Hinglish) |
| :---: | :--- | :--- |
| **01** | [01_the_chameleon_classifier_intuition.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/01_the_chameleon_classifier_intuition.md) | ML ka Girgit (Chameleon): Logistic Regression vs Linear SVM vs Perceptron, Big Data scaling wall, aur scaling kyu compulsory hai. |
| **02** | [02_loss_functions_and_decision_boundary_math.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/02_loss_functions_and_decision_boundary_math.md) | Margin math ($z = y \cdot f(x)$), 4 loss functions (`log_loss`, `hinge`, `modified_huber`, `perceptron`), aur regularization penalties. |
| **03** | [03_classification_metrics_and_threshold_tuning.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/03_classification_metrics_and_threshold_tuning.md) | Accuracy trap in imbalanced data, Confusion Matrix, Precision vs. Recall business tradeoffs, ROC-AUC, aur threshold tuning (0.50 vs 0.30). |
| **04** | [04_hyperparameter_tuning_gridsearch_strategy.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/04_hyperparameter_tuning_gridsearch_strategy.md) | Step 8: Multi-parameter tuning with `GridSearchCV` (`loss`, `penalty`, `alpha`, `learning_rate`, `class_weight='balanced'`). |
| **05** | [05_the_quad_classification_battle.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/05_the_quad_classification_battle.md) | Step 9: The Quad Battle (Batch Logistic Regression vs Linear SVC vs Perceptron vs SGDClassifier), latency vs accuracy tradeoffs. |
| **06** | [06_out_of_core_streaming_classification_with_partial_fit.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/06_out_of_core_streaming_classification_with_partial_fit.md) | 1 Crore transactions ko stream karna, `classes=np.unique(...)` ka golden rule, aur zero-memory streaming architecture. |
| **07** | [07_production_pipeline_and_live_inference.md](file:///home/python/03_Custom_addons/ML/memory_notes/01_Linear_Family/07_SGD_Classifier/07_production_pipeline_and_live_inference.md) | Step 11 & 12: Production pipeline packaging into `.joblib`, microsecond single-record probability inference, aur deployment checklist. |

---

## ⚡ 10-Second Interview Flashcard

| Question | Junior Developer Answer ❌ | Senior Developer Answer ✅ |
| :--- | :--- | :--- |
| **"SGDClassifier mein SVM aur Logistic Regression kaise switch karein?"** | Alag modules import karke. | Sirf `loss='hinge'` pass karne se SVM ban jata hai, aur `loss='log_loss'` pass karne se Logistic Regression ban jata hai! |
| **"Imbalanced dataset mein SGDClassifier kaise handle karein?"** | Data duplicate karke. | `class_weight='balanced'` parameter pass karne se minority class ke samples ko unke inverse frequency ke proportional weight mil jata hai, jisse recall instantly improve hoti hai! |
| **"`partial_fit()` mein `classes` argument kyu zaroori hota hai?"** | Sirf formality hai. | Model ko first batch par pata hona chahiye ki total classes kitni hain (e.g. [0, 1]). Agar pehle batch mein sirf 0 aayi, toh model bina classes argument ke crash ho jayega. |

---

## 💻 Master Copy-Paste Boilerplate: End-to-End SGD Classification (Steps 8-12)
*(Common Steps 1-7 chalaane ke baad is block ko paste karein)*

```python
# ==============================================================================
# ⚡ MASTER BOILERPLATE: SGDCLASSIFIER WITH BALANCED WEIGHTS & EARLY STOPPING
# ==============================================================================
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib

# 1. Initialize Best SGD Classifier
sgd_clf = SGDClassifier(
    loss='log_loss',
    penalty='l2',
    alpha=0.001,
    learning_rate='adaptive',
    eta0=0.01,
    class_weight='balanced',
    max_iter=5000,
    tol=1e-3,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    random_state=42
)

# 2. Fit on Scaled Train Data
sgd_clf.fit(X_train_final, y_tr)

# 3. Assemble Production Pipeline
prod_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('sgd_classifier', sgd_clf)
])
prod_pipeline.fit(X_tr, y_tr)

# 4. Serialize to Disk
joblib.dump(prod_pipeline, 'production_models/sgd_classifier_pipeline.joblib')
```
