# 📐 Splitting Criteria: Squared Error vs. Friedman MSE vs. MAE

- `criterion='squared_error'` (MSE): Standard variance reduction; sensitive to extreme target outliers.
- `criterion='friedman_mse'`: Friedman's improvement for gradient boosting improvements.
- `criterion='absolute_error'` (MAE): Uses median within leaves; robust against target outliers.
- `criterion='poisson'`: Models positive count targets (e.g. queue arrivals, insurance claim frequency).
