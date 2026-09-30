# 🛡️ The Variance Smoothing Guard (`var_smoothing`)

When a feature has near-zero variance within a class (e.g. constant readings or rare subgroups), division by $\sigma_{jc}^2 \approx 0$ causes numerical overflow.
Scikit-learn applies `var_smoothing` (default $10^{-9}$):

$$\sigma_{jc,\text{smoothed}}^2 = \sigma_{jc}^2 + \epsilon \cdot \max_{k, l}(\sigma_{kl}^2)$$

Where $\epsilon = \text{var\_smoothing}$.
- Setting `var_smoothing=1e-2` or `1e-3` regularizes sharp densities and prevents overconfidence on small samples.
