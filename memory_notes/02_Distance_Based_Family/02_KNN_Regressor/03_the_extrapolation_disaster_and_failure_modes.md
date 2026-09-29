# Guide 03: The Extrapolation Disaster & Critical Failure Modes

> **Note on Workflow:**
> For train/test split rules, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **most dangerous limitation of KNN Regression**.

---

## 1. The Extrapolation Disaster 🚨

**KNN CANNOT EXTRAPOLATE OUTSIDE THE RANGE OF ITS TRAINING DATA!**

This is the single biggest architectural limitation of distance-based regression.

### Concrete Example:
Suppose you train a house price model where square footage in the training data ranges from `500` sq ft to `3,500` sq ft, with prices from `\\$100,000` to `\\$700,000`.

Now, predict the price of a massive **15,000 sq ft Luxury Mansion**:
* **Linear Regression:** Extrapolates along the trendline:
  $$\\text{Price} = \\$200 \\times 15,000 = \\mathbf{\\$3,000,000} \\quad \\text{(Reasonable!)}$$
* **KNN Regressor:** Searches for the closest houses to 15,000 sq ft.
  The closest houses in the training set are the 3,500 sq ft homes!
  Their average price is $\\$700,000$.
  KNN predicts: **$\\$700,000$!** 💥

```
Price ^
      |                                              / (Linear Extrapolation)
      |                                            /
$700k | - - - - - - - - o o o o - - - - - - - - - - - (KNN Flatlines forever!)
      |               o
$100k |       o   o
      +-------|---------|----------------------------> SqFt
             500      3,500                        15,000 (Mansion)
              [ Training Domain ]
```

> **The Golden Law:**
> KNN Regressor predictions are strictly bounded by:
> $$\\min(y_{\\text{train}}) \\le \\hat{y} \\le \\max(y_{\\text{train}})$$
> It can **NEVER** predict a value higher than the maximum training label or lower than the minimum training label!

---

## 2. Other Failure Modes

1. **High Dimensionality ($D > 20$):**
   * As dimensions grow, volume expands exponentially and all points become equidistant.
   * Nearest neighbors become random points, causing predictions to converge to the global mean.
2. **Unscaled Features:**
   * A feature with scale $0 - 100,000$ will overpower features with scale $0 - 1$, causing 99% of distance to depend on that single feature.
3. **Slow Inference on Big Data:**
   * Scoring 100,000 test points against 1,000,000 training points takes billions of operations without specialized indexes (KD-Tree / Ball-Tree).
