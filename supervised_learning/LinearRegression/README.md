# 📈 Linear Regression: Projects & Production Notebooks Suite

Yeh folder **Linear Regression (Ordinary Least Squares - OLS)** ke saare end-to-end practical machine learning projects ka hub hai. Yahan har ek notebook raw data loading se lekar model evaluation aur serialization tak structured format mein ready hai.

---

## 🗂️ Project Directory Structure

```text
LinearRegression/
├── README.md                                          # Projects Portfolio Guide (Yeh file)
│
├── 📓 Notebooks:
│   ├── 01_Salary_Prediction.ipynb (Salary_Prediction.ipynb)
│   ├── 02_Advertising_Sales.ipynb (Advertising_Sales.ipynb)
│   ├── 03_Student_Performance_Prediction.ipynb
│   ├── 04_Electricity_Consumption.ipynb
│   ├── 05_Rent_Price_Prediction.ipynb
│   ├── 06_Car_Price_Prediction.ipynb
│   └── 07_House_Price_Prediction.ipynb
│
├── 📂 data/                                           # Clean Structured Datasets
│   ├── salary_data/                                   # Salary_Data.csv
│   ├── advertising_sales/                             # Advertising Budget and Sales.csv
│   ├── student_performance/                           # Students Performance .csv
│   ├── electricity_consumption/                       # Hospital Building Dataset.xlsx
│   ├── house_rent/                                    # House_Rent_Dataset.csv
│   ├── used_car/                                      # UsedCarInfo.csv
│   └── house_price/                                   # Kaggle House Prices (train.csv, test.csv)
│
└── 📦 production_models/                              # Serialized Trained Model Artifacts
    ├── salary_predictor_model.pkl
    ├── advertising_sales_model.pkl
    ├── student_gpa_model.pkl
    ├── hospital_electricity_model.pkl
    ├── house_rent_model.pkl
    └── car_price_model.pkl
```

---

## 📊 Summary of Projects

| Project / Notebook | Target Variable ($y$) | Input Data Format | Primary Evaluation Metric | Trained Artifact |
| :--- | :--- | :--- | :---: | :--- |
| [Salary_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Salary_Prediction.ipynb) | `Salary` | `data/salary_data/Salary_Data.csv` | $R^2 \approx 0.95$ | `salary_predictor_model.pkl` |
| [Advertising_Sales.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Advertising_Sales.ipynb) | `Sales` | `data/advertising_sales/Advertising Budget and Sales.csv` | $R^2 \approx 0.89$ | `advertising_sales_model.pkl` |
| [Student_Performance_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Student_Performance_Prediction.ipynb) | `GPA / Score` | `data/student_performance/Students Performance .csv` | $R^2 \approx 0.88$ | `student_gpa_model.pkl` |
| [Electricity_Consumption.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Electricity_Consumption.ipynb) | `Energy Consumption` | `data/electricity_consumption/Hospital Building Dataset.xlsx` | $R^2 \approx 0.82$ | `hospital_electricity_model.pkl` |
| [Rent_Price_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Rent_Price_Prediction.ipynb) | `Rent` | `data/house_rent/House_Rent_Dataset.csv` | RMSE, $R^2$ | `house_rent_model.pkl` |
| [Car_Price_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/Car_Price_Prediction.ipynb) | `Price` | `data/used_car/UsedCarInfo.csv` | RMSE, $R^2$ | `car_price_model.pkl` |
| [House_Price_Prediction.ipynb](file:///home/python/03_Custom_addons/ML/supervised_learning/LinearRegression/House_Price_Prediction.ipynb) | `SalePrice` | `data/house_price/train.csv` | Log-RMSE, $R^2$ | Kaggle Submission CSV |

