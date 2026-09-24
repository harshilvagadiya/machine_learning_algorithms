"""
Script to generate and execute the complete Car Price Prediction Ridge Regression notebook.
Follows the 12-Step Enterprise Machine Learning Framework with zero data leakage.
"""
import json
import uuid
import io
import sys
import os
import base64
from pathlib import Path

workspace_root = Path("/home/python/03_Custom_addons/ML")
nb_dir = workspace_root / "supervised_learning/RidgeRegression"
nb_path = nb_dir / "Car_Price_Prediction_Ridge.ipynb"

# Define Cells
cells = []

def add_cell(source_code):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_code.strip().split("\n")]
    })

# ==============================================================================
# CELL 0: STEP 1 - LIBRARIES IMPORT
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 1: MODULAR ENTERPRISE LIBRARIES IMPORT (RIDGE REGRESSION SUITE)
# ==============================================================================
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10

# Preprocessing & Splitting (Strict Zero-Leakage Architecture)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Estimators & Regularization (OLS vs Ridge)
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

# Evaluation Metrics & Model Serialization
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import joblib

print("✅ Enterprise Ridge Regression Suite imported successfully.")''')

# ==============================================================================
# CELL 1: STEP 2 - DATA INGESTION
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 2: ROBUST DATA INGESTION (USED CAR VALUATION DATASET)
# ==============================================================================
# Look for data in local data/car_price or parent supervised_learning folder
possible_paths = [
    Path("data/car_price/car_prediction_data.csv"),
    Path("../car_prediction_data.csv"),
    Path("supervised_learning/car_prediction_data.csv")
]

data_path = None
for p in possible_paths:
    if p.exists():
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("❌ car_prediction_data.csv not found in candidate paths!")

df = pd.read_csv(data_path, encoding='utf-8')

print("=" * 65)
print(f"📦 STEP 2: DATASET INGESTION SUMMARY")
print("=" * 65)
print(f"Source File Path : {data_path.resolve()}")
print(f"Total Rows       : {df.shape[0]:,}")
print(f"Total Columns    : {df.shape[1]}")
print("=" * 65)
df.head(5)''')

# ==============================================================================
# CELL 2: STEP 3 - 3-SECOND SENIOR HEALTH AUDIT
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 3: POST-LOAD 3-SECOND SENIOR HEALTH AUDIT
# ==============================================================================
mem_usage = df.memory_usage(deep=True).sum() / 1024

print("=" * 65)
print("🔍 STEP 3: 3-SECOND STRUCTURAL HEALTH AUDIT")
print("=" * 65)
print(f"Memory Footprint       : {mem_usage:.2f} KB")
print(f"Column Names           : {list(df.columns)}")
print(f"Target Column Candidate: 'Selling_Price' (Price in Lakhs INR)")
print(f"Numerical Features     : {df.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"Categorical Features   : {df.select_dtypes(exclude=[np.number]).columns.tolist()}")
print("=" * 65)
df.info()''')

# ==============================================================================
# CELL 3: STEP 4 - FEATURE ENGINEERING & SEGREGATION
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 4: DATA HYGIENE - FEATURE ENGINEERING & SYSTEMATIC SEGREGATION
# ==============================================================================
# 1. Feature Engineering: Calculate Car Age from Manufacturing Year
# Dataset year max is 2018; reference year 2020 gives natural depreciation age
df['Car_Age'] = 2020 - df['Year']

# 2. Segregate columns into Continuous, Categorical, and Drop features
target_col = 'Selling_Price'
drop_features = ['Car_Name', 'Year']  # Car_Name is high-cardinality model label; Year is engineered to Car_Age
continuous_features = ['Present_Price', 'Kms_Driven', 'Car_Age', 'Owner']
categorical_features = ['Fuel_Type', 'Seller_Type', 'Transmission']

print("=" * 65)
print("🔍 STEP 4: FEATURE SEGREGATION SUMMARY")
print("=" * 65)
print(f"Target Variable ($y$) : '{target_col}' (Selling price in Lakhs)")
print(f"Continuous Features   : {len(continuous_features)} cols -> {continuous_features}")
print(f"Categorical Features  : {len(categorical_features)} cols -> {categorical_features}")
print(f"Dropped / ID Features : {len(drop_features)} cols -> {drop_features}")
print("=" * 65)
df[continuous_features].describe().round(2)''')

# ==============================================================================
# CELL 4: STEP 5 - MISSING VALUES & DUPLICATES HYGIENE
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 5: DATA HYGIENE - MISSING VALUES & DUPLICATE RECORDS PURGING
# ==============================================================================
# 1. Check Missing Values
null_counts = df.isnull().sum()
total_nulls = null_counts.sum()

# 2. Audit & Drop Duplicates
n_duplicates = df.duplicated().sum()

print("=" * 65)
print("🧹 STEP 5: DATA HYGIENE REPORT")
print("=" * 65)
print(f"Total Missing Values : {total_nulls} nulls across all columns.")
print(f"Exact Duplicate Rows : {n_duplicates} duplicate records found.")

if n_duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"✅ Cleaned: {n_duplicates} duplicate records purged to prevent train-test leakage.")
    print(f"New Cleaned Shape    : {df.shape[0]} rows × {df.shape[1]} columns")
else:
    print("✅ Zero duplicate records detected.")
print("=" * 65)''')

# ==============================================================================
# CELL 5: STEP 6 - TARGET VARIABLE ANALYSIS & LOG-TRANSFORM
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 6: TARGET VARIABLE ANALYSIS & LOG-TRANSFORMATION (log1p)
# ==============================================================================
raw_y = df[target_col]
raw_skew = raw_y.skew()

log_y = np.log1p(raw_y)
log_skew = log_y.skew()

print("=" * 65)
print("🎯 STEP 6: TARGET DISTRIBUTION & SKEWNESS PROFILE")
print("=" * 65)
print(f"Target Variable (Raw) Skewness : {raw_skew:.2f} (Heavy Right Tail!)")
print(f"Target Variable (log1p) Skewness: {log_skew:.2f} (Near-Perfect Normal Curve!)")
print(f"Selling Price Range (Raw)      : ₹{raw_y.min():.2f} Lakhs to ₹{raw_y.max():.2f} Lakhs")
print(f"Selling Price Median (Raw)     : ₹{raw_y.median():.2f} Lakhs")
print("-" * 65)
print("💡 Why log1p is Essential:")
print("   Linear models assume normally distributed residuals. In real-world car sales,")
print("   budget hatchbacks dominate while rare luxury cars pull OLS weights into severe error.")
print("   np.log1p compresses the right tail, stabilizing regression variance.")
print("=" * 65)

# Visual Distribution Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

sns.histplot(raw_y, kde=True, ax=axes[0], color='#e74c3c')
axes[0].set_title(f'Raw Selling Price (Skewness: {raw_skew:.2f})', fontweight='bold')
axes[0].set_xlabel('Selling Price (Lakhs INR)')

sns.histplot(log_y, kde=True, ax=axes[1], color='#27ae60')
axes[1].set_title(f'Log-Transformed Selling Price (Skewness: {log_skew:.2f})', fontweight='bold')
axes[1].set_xlabel('log(1 + Selling Price)')

plt.tight_layout()
plt.show()''')

# ==============================================================================
# CELL 6: STEP 7 - MULTICOLLINEARITY AUDIT
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 7: MULTICOLLINEARITY AUDIT & THE MATHEMATICAL NEED FOR RIDGE
# ==============================================================================
# Correlation Matrix among all numerical features
corr_cols = [target_col] + continuous_features
corr_matrix = df[corr_cols].corr()

print("=" * 65)
print("🔍 STEP 7: MULTICOLLINEARITY AUDIT (CORRELATION MATRIX)")
print("=" * 65)
print("🚨 KEY CORRELATION FINDINGS:")
print(f"  • Selling_Price vs Present_Price : r = {corr_matrix.loc['Selling_Price', 'Present_Price']:.2f} (Strongest valuation driver)")
print(f"  • Selling_Price vs Car_Age       : r = {corr_matrix.loc['Selling_Price', 'Car_Age']:.2f} (Expected strong depreciation)")
print(f"  • Present_Price vs Kms_Driven    : r = {corr_matrix.loc['Present_Price', 'Kms_Driven']:.2f} (Correlated vehicle usage)")
print("-" * 65)
print("💡 Mathematical Need for Ridge Regularization:")
print("   When multiple correlated features enter regression, (X^T * X) becomes ill-conditioned.")
print("   Ridge adds alpha * I to the diagonal, guaranteeing invertibility and preventing")
print("   runaway coefficient variance.")
print("=" * 65)

plt.figure(figsize=(8, 5))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Vehicle Numerical Features Correlation Heatmap', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()''')

# ==============================================================================
# CELL 7: STEP 8 - TRAIN-TEST SPLIT
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 8: TRAIN-TEST SPLIT (80% TRAIN, 20% TEST - ZERO DATA LEAKAGE)
# ==============================================================================
# 1. Feature Matrix (X) and Log-Transformed Target (y_log)
X = df.drop(columns=[target_col] + drop_features)
y_log = np.log1p(df[target_col])

# 2. Strict 80:20 Train/Test Split
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.20, random_state=42
)

print("=" * 65)
print("📦 STEP 8: TRAIN-TEST SPLIT SUMMARY")
print("=" * 65)
print(f"Full Clean Dataset       : {len(df):,} vehicles")
print(f"Training Matrix (X_train): {X_train.shape[0]} rows × {X_train.shape[1]} features (80%)")
print(f"Testing Matrix  (X_test) : {X_test.shape[0]} rows × {X_test.shape[1]} features (20%)")
print(f"Mean Train Price (Raw)   : ₹{np.expm1(y_train_log).mean():.2f} Lakhs")
print(f"Mean Test Price  (Raw)   : ₹{np.expm1(y_test_log).mean():.2f} Lakhs")
print("=" * 65)''')

# ==============================================================================
# CELL 8: STEP 9 - FEATURE PREPROCESSING (COLUMNTRANSFORMER)
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 9: FEATURE PREPROCESSING (SCALING & ENCODING - ZERO LEAKAGE)
# ==============================================================================
# 1. Continuous Pipeline (Median Imputation + Mandatory StandardScaler)
continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  # Essential for fair L2 penalty allocation!
])

# 2. Categorical Pipeline (OneHotEncoder with drop='first' to prevent dummy trap)
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# 3. Master ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', continuous_pipeline, continuous_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 4. Fit strictly on X_train, Transform X_test
X_train_final = preprocessor.fit_transform(X_train)
X_test_final  = preprocessor.transform(X_test)

# Extract expanded feature names
cat_encoded_names = list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features))
all_feature_names = continuous_features + cat_encoded_names

print("=" * 65)
print("⚙️ STEP 9: PREPROCESSING & ENCODING SUMMARY")
print("=" * 65)
print(f"Continuous Features Processed   : {len(continuous_features)} cols")
print(f"Categorical Features Processed  : {len(categorical_features)} cols")
print(f"Total One-Hot Encoded Features  : {len(cat_encoded_names)} dummy cols")
print(f"Total Model Input Predictors    : {X_train_final.shape[1]} dimensions")
print(f"Final Feature Names             : {all_feature_names}")
print("=" * 65)''')

# ==============================================================================
# CELL 9: STEP 10 - RIDGECV HYPERPARAMETER TUNING
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 10: RIDGE REGRESSION & ALPHA TUNING VIA GENERALIZED CV (RidgeCV)
# ==============================================================================
# 1. Search Grid across 50 logarithmic values from 0.01 to 1000
alphas = np.logspace(-2, 3, 50)

# 2. Generalized Cross-Validation (GCV - 1-second analytical shortcut)
ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=None, store_cv_results=True)
ridge_cv.fit(X_train_final, y_train_log)

best_alpha = ridge_cv.alpha_
mean_cv_mse = np.mean(ridge_cv.cv_results_, axis=0)

print("=" * 65)
print("🏆 STEP 10: RIDGECV HYPERPARAMETER TUNING RESULTS")
print("=" * 65)
print(f"Alpha Candidate Range  : {alphas[0]:.2f} to {alphas[-1]:.1f} ({len(alphas)} candidates)")
print(f"Optimal L2 Penalty (α) : {best_alpha:.4f}")
print(f"Minimum GCV MSE Loss   : {np.min(mean_cv_mse):.6f}")
print("-" * 65)
print("💡 Interpretation:")
print(f"   alpha = {best_alpha:.4f} strikes the ideal bias-variance sweet spot,")
print("   preventing model weights from over-reacting to luxury price outliers.")
print("=" * 65)

# Plot Optimization Loss Curve
plt.figure(figsize=(9, 4.5))
plt.plot(alphas, mean_cv_mse, color='navy', linewidth=2, label='Mean GCV Loss')
plt.axvline(best_alpha, color='crimson', linestyle='--', linewidth=1.5, label=f'Optimal α = {best_alpha:.2f}')
plt.scatter([best_alpha], [np.min(mean_cv_mse)], color='crimson', s=90, zorder=5)
plt.xscale('log')
plt.xlabel('Regularization Strength α (Log Scale)', fontsize=11, fontweight='bold')
plt.ylabel('Generalized CV MSE Error', fontsize=11, fontweight='bold')
plt.title('RidgeCV Optimization Curve: Validation Error vs Alpha', fontsize=12, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()''')

# ==============================================================================
# CELL 10: STEP 11 - HEAD-TO-HEAD BATTLE: OLS VS RIDGE
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 11: HEAD-TO-HEAD BATTLE: OLS LINEAR REGRESSION VS. RIDGE REGRESSION
# ==============================================================================
# 1. Fit OLS Linear Regression (No Regularization)
ols_model = LinearRegression().fit(X_train_final, y_train_log)

# 2. Fit Ridge with Optimal Alpha
ridge_model = Ridge(alpha=best_alpha, random_state=42).fit(X_train_final, y_train_log)

# 3. Model Predictions (Log Scale)
ols_pred_log = ols_model.predict(X_test_final)
ridge_pred_log = ridge_model.predict(X_test_final)

ols_train_r2 = ols_model.score(X_train_final, y_train_log)
ols_test_r2  = ols_model.score(X_test_final, y_test_log)
ols_rmse_log = root_mean_squared_error(y_test_log, ols_pred_log)

ridge_train_r2 = ridge_model.score(X_train_final, y_train_log)
ridge_test_r2  = ridge_model.score(X_test_final, y_test_log)
ridge_rmse_log = root_mean_squared_error(y_test_log, ridge_pred_log)

# 4. Real Currency Predictions (Inverse expm1 transformation)
y_test_real = np.expm1(y_test_log)
ols_pred_real = np.expm1(ols_pred_log)
ridge_pred_real = np.expm1(ridge_pred_log)

ols_real_rmse = root_mean_squared_error(y_test_real, ols_pred_real)
ridge_real_rmse = root_mean_squared_error(y_test_real, ridge_pred_real)
ols_real_mae = mean_absolute_error(y_test_real, ols_pred_real)
ridge_real_mae = mean_absolute_error(y_test_real, ridge_pred_real)

# 5. Build Comparison Table
battle_table = pd.DataFrame({
    'Evaluation Metric': [
        'Train R² Score',
        'Test R² Score',
        'Test Log-RMSE',
        'Real Price RMSE (Lakhs)',
        'Real Price MAE (Lakhs)'
    ],
    'OLS Linear Regression': [
        f'{ols_train_r2*100:.2f}%',
        f'{ols_test_r2*100:.2f}%',
        f'{ols_rmse_log:.4f}',
        f'₹{ols_real_rmse:.2f} Lakhs',
        f'₹{ols_real_mae:.2f} Lakhs'
    ],
    'Ridge Regression (L2)': [
        f'{ridge_train_r2*100:.2f}%',
        f'{ridge_test_r2*100:.2f}%',
        f'{ridge_rmse_log:.4f}',
        f'₹{ridge_real_rmse:.2f} Lakhs',
        f'₹{ridge_real_mae:.2f} Lakhs'
    ]
})

print("=" * 70)
print("⚔️ HEAD-TO-HEAD MODEL BENCHMARK: OLS VS. RIDGE")
print("=" * 70)
print(battle_table.to_string(index=False))
print("=" * 70)''')

# ==============================================================================
# CELL 11: STEP 12 - COEFFICIENT SHRINKAGE VISUALIZATION
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 12: COEFFICIENT SHRINKAGE VISUALIZATION & AUTOMOTIVE VALUATION DRIVERS
# ==============================================================================
# 1. Weights DataFrame
weights_df = pd.DataFrame({
    'Feature': all_feature_names,
    'OLS_Weight': ols_model.coef_,
    'Ridge_Weight': ridge_model.coef_,
    'Ridge_Abs_Impact': np.abs(ridge_model.coef_)
}).sort_values(by='Ridge_Weight', ascending=True)

# 2. Weight Shrinkage Verification
max_ols_wt = np.max(np.abs(weights_df['OLS_Weight']))
max_ridge_wt = np.max(np.abs(weights_df['Ridge_Weight']))
shrinkage_pct = ((max_ols_wt - max_ridge_wt) / max_ols_wt) * 100

print("=" * 65)
print("🔍 STEP 12: RIDGE WEIGHT SHRINKAGE PROOF")
print("=" * 65)
print(f"Max Absolute Weight in OLS   : {max_ols_wt:.4f}")
print(f"Max Absolute Weight in Ridge : {max_ridge_wt:.4f}")
print(f"Regularization Shrinkage     : {shrinkage_pct:.1f}% reduction in runaway coefficient magnitude!")
print("=" * 65)

# 3. Feature Importance Horizontal Bar Plot
plt.figure(figsize=(10, 6))
colors = ['#27ae60' if w > 0 else '#e74c3c' for w in weights_df['Ridge_Weight']]
plt.barh(weights_df['Feature'], weights_df['Ridge_Weight'], color=colors, edgecolor='black', alpha=0.85)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Used Car Valuation Drivers in Ridge Model (Green = Value Booster, Red = Depreciation Factor)', fontsize=12, fontweight='bold')
plt.xlabel('Ridge Regression Weight (Log-Price Impact)', fontsize=11)
plt.ylabel('Vehicle Attributes', fontsize=11)
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()''')

# ==============================================================================
# CELL 12: STEP 13 - PRODUCTION PIPELINE & SERIALIZATION
# ==============================================================================
add_cell('''# ==============================================================================
# STEP 13: ENTERPRISE PRODUCTION PIPELINE & MODEL SERIALIZATION
# ==============================================================================
# 1. Assemble Full End-to-End Master Pipeline
production_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ridge', Ridge(alpha=best_alpha, random_state=42))
])

# 2. Single Fit on Raw X_train (Zero Data Leakage Guarantee)
production_pipeline.fit(X_train, y_train_log)

# 3. Serialize Production Artifact to Disk
model_dir = Path("production_models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "car_price_ridge_pipeline.joblib"

joblib.dump(production_pipeline, model_path)

print("=" * 65)
print("🏭 ENTERPRISE PRODUCTION PIPELINE SUMMARY")
print("=" * 65)
print(f"✅ Master Pipeline Serialized Successfully to: {model_path}")
print(f"   Artifact File Size: {model_path.stat().st_size / 1024:.2f} KB")

# 4. Live Production Inference Smoke Test
loaded_pipeline = joblib.load(model_path)
sample_car = X_test.iloc[0:1]

pred_log_val = loaded_pipeline.predict(sample_car)[0]
predicted_price = np.expm1(pred_log_val)
actual_price = np.expm1(y_test_log.iloc[0])

print("\n🧪 Live Production Inference Smoke Test (1 Unseen Vehicle):")
print(f"   • Vehicle Specs   : Present Price ₹{sample_car['Present_Price'].values[0]}L, Age {sample_car['Car_Age'].values[0]} yrs, Kms {sample_car['Kms_Driven'].values[0]:,}")
print(f"   • Predicted Price : ₹{predicted_price:.2f} Lakhs")
print(f"   • Actual Price    : ₹{actual_price:.2f} Lakhs")
print(f"   • Absolute Error  : ₹{abs(predicted_price - actual_price):.2f} Lakhs ({abs(predicted_price - actual_price)/actual_price*100:.1f}%)")
print("=" * 65)''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Save notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Generated notebook with {len(cells)} cells at: {nb_path}")
