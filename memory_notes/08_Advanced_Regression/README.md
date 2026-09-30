# 📈 Advanced Regression Knowledge Hub

## Included Paradigms & Algorithm Families

1. **[Generalized Linear Models (GLM)](Generalized_Linear_Models.md)**
   - Poisson Regression ($\mu = \operatorname{Var}$)
   - Gamma Regression ($V(\mu) = \mu^2$)
   - Tweedie Regression ($V(\mu) = \mu^p$)
   - **Negative Binomial Regression** ($V(\mu) = \mu + \alpha \mu^2$ for severe overdispersion)
2. **[Robust Regression](Robust_Regression.md)**
   - Huber Regressor ($L_1/L_2$ hybrid penalty)
   - RANSAC (Random Sample Consensus for extreme adversarial contamination)
   - Theil-Sen (Non-parametric median of pairwise slopes, breakdown point 29.3%)
3. **[Polynomial Regression](Polynomial_Regression.md)**
   - Non-linear basis expansion, Vandermonde matrices, Runge's phenomenon mitigation.
4. **[Quantile Regression](Quantile_Regression.md)**
   - Pinball loss optimization, conditional quantile intervals, heteroscedastic uncertainty bounds.
5. **[Gaussian Process Regression (GPR)](Gaussian_Process_Regression.md)**
   - Non-parametric Bayesian kernel modeling, exact closed-form predictive mean & epistemic variance.
