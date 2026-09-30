# 🎛️ Tuning Alpha & Class Imbalance

Tune additive smoothing parameter $\alpha \in [0.1, 1.0, 5.0]$.
To account for class imbalance (e.g. rare network intrusions $0.1\%$), adjust class priors via `class_prior=[0.999, 0.001]` or use threshold tuning on `predict_proba()`.
