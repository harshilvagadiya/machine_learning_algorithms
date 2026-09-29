# Guide 02: Uniform vs Distance-Weighted Surfaces

> **Note on Workflow:**
> For baseline Scaling/Encoding transforms, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **prediction formulas and regression surface shapes**.

---

## 1. The Two Weighting Modes in `KNeighborsRegressor`

In `KNeighborsRegressor(weights=...)`, there are two choices:

### 1.1 `weights='uniform'` (The Simple Average)
All $K$ nearest neighbors have an equal say, regardless of how close they are:

$$\\hat{y}(\\mathbf{x}^*) = \\frac{1}{K} \\sum_{i \\in \\mathcal{N}_K(\\mathbf{x}^*)} y_i$$

* **Geometric Behavior:** Produces a **Piecewise Constant Step Function**!
* As your query point moves across feature space, the set of $K$ neighbors remains identical, producing a flat horizontal prediction.
* The moment a new neighbor enters the $K$-ball, the prediction abruptly **jumps like a staircase**.

```
Target y ^
         |              +---------+ (Step jump!)
         |    +---------+
         |----+
         +-------------------------------------> Feature x
                 Uniform KNN: Jagged Staircase
```

---

### 1.2 `weights='distance'` (Inverse Distance Interpolation)
Each neighbor's contribution is weighted by the inverse of its distance:

$$w_i = \\frac{1}{d(\\mathbf{x}^*, \\mathbf{x}_i)}$$

$$\\hat{y}(\\mathbf{x}^*) = \\frac{\\sum_{i \\in \\mathcal{N}_K} w_i y_i}{\\sum_{i \\in \\mathcal{N}_K} w_i}$$

* **Geometric Behavior:** Produces a **Smooth, Continuous Interpolating Curve**!
* Neighbors that are extremely close to $\\mathbf{x}^*$ dominate the prediction.
* Eliminates the sharp staircase jumps, creating an organic, flowing regression curve.

```
Target y ^
         |                 _ . - - . _
         |             . '             ' .
         |      _ . '
         +-------------------------------------> Feature x
              Distance-Weighted KNN: Smooth Curve
```

---

## 2. When to Use Which?

* **Use `weights='distance'` (Recommended for most regression tasks):**
  * When you want smooth continuous predictions (e.g. house prices, temperatures, stock volatility).
  * When data points are unevenly distributed across feature space.
* **Use `weights='uniform'`:**
  * When individual labels are very noisy, and you want an unweighted average to cancel out random noise.
