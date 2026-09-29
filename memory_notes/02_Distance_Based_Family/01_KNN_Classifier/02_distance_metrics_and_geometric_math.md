# Guide 02: Distance Metrics & Geometric Mathematics

> **Note on Workflow:**
> For baseline Scaling/Encoding transforms, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **mathematical distance metrics and why distance geometry dictates scaling**.

---

## 1. Minkowski Distance Family (The Foundation)

Most continuous distance metrics are special cases of the **Minkowski Distance** of order $p$:

$$D_{\text{Minkowski}}(\mathbf{u}, \mathbf{v}) = \left( \sum_{j=1}^D |u_j - v_j|^p \right)^{1/p}$$

In `scikit-learn`: `KNeighborsClassifier(metric='minkowski', p=...)`

### 1.1 Manhattan Distance ($p = 1$, $L_1$ Norm)
$$D_{\text{Manhattan}}(\mathbf{u}, \mathbf{v}) = \sum_{j=1}^D |u_j - v_j|$$
* **Intuition:** City-block distance. You can only move along axis-aligned grid lines (like Manhattan city streets).
* **Best Used For:** High-dimensional spaces, sparse data, or when features have varying units and extreme outliers (less sensitive to squared outlier penalties).

### 1.2 Euclidean Distance ($p = 2$, $L_2$ Norm — Default)
$$D_{\text{Euclidean}}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{j=1}^D (u_j - v_j)^2}$$
* **Intuition:** Straight line "as the crow flies".
* **Geometric Behavior:** Isotropic (spherical contours). Penalizes larger discrepancies much more heavily due to the squaring term.

### 1.3 Chebyshev Distance ($p \to \infty$, $L_\infty$ Norm)
$$D_{\text{Chebyshev}}(\mathbf{u}, \mathbf{v}) = \max_{j=1 \dots D} |u_j - v_j|$$
* **Intuition:** Chessboard distance (king moves). Only the single worst coordinate difference counts.

---

## 2. Categorical & Specialized Metrics

### 2.1 Hamming Distance (Binary / Categorical)
$$D_{\text{Hamming}}(\mathbf{u}, \mathbf{v}) = \frac{1}{D} \sum_{j=1}^D \mathbb{I}(u_j \neq v_j)$$
* Proportion of positions in which two vectors differ.
* Crucial when working with one-hot encoded or binary boolean attributes.

### 2.2 Cosine Distance (Direction over Magnitude)
$$D_{\text{Cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
* Measures the angle between two vectors, ignoring their physical length.
* Commonly used for text classification (TF-IDF), gene expression profiles, and recommendation embeddings.

---

## 3. Why Feature Scaling is 100% NON-NEGOTIABLE in KNN

In linear models, unscaled features result in skewed gradient descent convergence or distorted regularized weights ($\beta$).
**In KNN, unscaled features completely DESTROY the algorithm's validity!**

### Concrete Disaster Example:
Suppose we classify credit risk using 2 features:
1. `Annual Income`: $20,000 to $200,000
2. `Age`: $18$ to $70$ years

Compute Euclidean distance between Person A `($50,000, 25)` and Person B `($51,000, 60)`:
$$d = \sqrt{(51,000 - 50,000)^2 + (60 - 25)^2} = \sqrt{1,000,000 + 1,225} = \sqrt{1,001,225} \approx 1,000.61$$

* Income difference ($1000^2 = 1,000,000$) accounted for **99.88%** of the distance.
* A massive age difference of **35 years** ($35^2 = 1,225$) accounted for only **0.12%**!
* **Result:** The model behaves as if `Age` does not even exist.

```
       Unscaled Space:                      Scaled Space (StandardScaler):
       Income ^                             z_Income ^
              |                                      |   o (B)
              |  o (B)                               |
              |                                      |       o (A)
              |  o (A)                               +-------------> z_Age
              +-------------> Age            Both features have equal geometric
       Age axis is flattened to zero!        variance & balanced voting power.
```

### Which Scaler to Use?
* **`StandardScaler` ($\mu=0, \sigma=1$):** Default recommendation. Preserves Gaussian properties and handles moderate variations well.
* **`MinMaxScaler` ($[0, 1]$):** Recommended when features have bounded physical ranges or when combining with sparse matrices.
* **`RobustScaler` (Median & IQR):** Mandatory if the dataset contains severe outliers, preventing extreme values from compressing legitimate variance.
