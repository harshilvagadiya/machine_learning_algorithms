# 🎯 Logistic Regression: Projects & Production Notebooks Suite

Yeh folder **Logistic Regression (Binary & Multiclass Classification)** ke saare end-to-end practical machine learning projects ka hub hai. Yahan har ek notebook 13-step standard enterprise framework follow karti hai: data hygiene, zero-leakage pre-processing, baseline benchmarking, odds ratio explainability, aur production pipeline serialization.

---

## 🗂️ Project Directory Structure

```text
LogisticRegression/
├── README.md                                          # Projects Portfolio Guide (Yeh file)
│
├── 📓 Notebooks:
│   ├── Student_Pass_Fail_Prediction.ipynb             # 🎓 Academic Performance Binary Classification
│   ├── Heart_Disease_Prediction.ipynb                 # 🫀 Clinical Multi-Hospital Disease Diagnosis
│   ├── Spam_Email_Detection.ipynb                     # 📧 NLP Text Classification (TF-IDF + Logistic)
│   └── Credit_Card_Fraud_Detection.ipynb              # 💳 Financial Risk & Application Approval
│
├── 📂 data/                                           # Clean Structured Datasets
│   ├── student_pass_fail_prediction/                  # student-por-train.csv (649 rows × 32 cols)
│   ├── heart_disease/ (heart+disease/)                # heart.csv, heart_cleveland.csv, multi-center
│   ├── spam_email_detection/                          # spam.csv (5,572 SMS/Emails)
│   └── credit_card_fraud_detection/                   # Credit_Card_Applications.csv
│
└── 📦 production_models/                              # Serialized Production Artifacts (.joblib / .pkl)
    ├── student_pass_fail_pipeline.joblib              # Full ColumnTransformer + Logistic Pipeline
    ├── heart_disease_classifier_pipeline.joblib       # Clinical Diagnostic Model Artifact
    └── spam_classifier_pipeline.pkl                   # TF-IDF + Logistic Classifier Pipeline
```

---

## 📊 Summary of Projects

| Project / Notebook | Target Variable ($y$) | Dataset Location | Baseline Benchmark | Model Performance | Serialized Artifact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| [Student_Pass_Fail_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LogisticRegression/Student_Pass_Fail_Prediction.ipynb) | `pass` (1=Pass, 0=Fail) | `data/student_pass_fail_prediction/student-por-train.csv` | **84.59%** (Zero-Rule) | **Test Acc: 96.92%**<br>ROC-AUC: **98.59%** | `student_pass_fail_pipeline.joblib` |
| [Heart_Disease_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LogisticRegression/Heart_Disease_Prediction.ipynb) | `target` (0=Normal, 1=Disease) | `data/heart_disease/heart.csv` | **54.46%** (Balanced) | **Test Acc: ~85.2%**<br>ROC-AUC: **91.4%** | `heart_disease_classifier_pipeline.joblib` |
| [Spam_Email_Detection.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LogisticRegression/Spam_Email_Detection.ipynb) | `v1` (ham / spam) | `data/spam_email_detection/spam.csv` | **86.60%** (Majority Ham) | **Test Acc: ~98.1%**<br>Precision: **100%** | `spam_classifier_pipeline.pkl` |
| [Credit_Card_Fraud_Detection.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LogisticRegression/Credit_Card_Fraud_Detection.ipynb) | `Class` (0=Denied, 1=Approved) | `data/credit_card_fraud_detection/Credit_Card_Applications.csv` | Imbalance Baseline | F1-Score, ROC-AUC | Production Suite |

