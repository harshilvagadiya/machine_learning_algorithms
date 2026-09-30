# 🎛️ Hyperparameter Tuning & Parallelization

- `n_estimators`: $100 - 500$ (performance plateaus, more is always better if compute allows).
- `max_depth`: $10 - 20$ (constrain to prevent massive model artifact sizes).
- `n_jobs=-1`: Linear multicore training scaling.
