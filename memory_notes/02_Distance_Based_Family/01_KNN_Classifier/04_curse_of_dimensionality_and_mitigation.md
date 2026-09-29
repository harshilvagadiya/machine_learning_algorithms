# Guide 04: The Curse of Dimensionality & Mitigation

> **Note on Workflow:**
> For general Feature Selection or Train/Test splits, refer to [Universal Pre-Flight Guide](../../00_Common_ML_Workflow/README.md).
> This document details the **mathematical breakdown of why high dimensions break distance metrics and how to fix it**.

---

## 1. What is the "Curse of Dimensionality"?

In 2D space ($D=2$), a circle has clear inner and outer areas. Points cluster visibly.
As the number of dimensions $D$ increases from 2 to 50, 100, or 10,000, geometry behaves counter-intuitively:

### 1.1 Exponential Volume Expansion
To capture a small neighborhood covering just $10\%$ ($0.1$) of the data space along each axis:
* In 1D: edge length = $0.1$ ($10\%$ of axis)
* In 2D: edge length = $\sqrt{0.1} \approx 0.316$ ($32\%$ of axis)
* In 10D: edge length = $0.1^{1/10} \approx 0.794$ ($79\%$ of axis)
* In 50D: edge length = $0.1^{1/50} \approx 0.955$ ($95.5\%$ of axis)

> **Conclusion:** In high dimensions, to find your "nearest 5 neighbors", you must expand your search window across almost the **entire length of the space**!

---

### 1.2 The Equidistance Phenomenon
As $D \to \infty$, the distance between the closest neighbor $d_{\min}$ and the farthest neighbor $d_{\max}$ becomes practically identical relative to the distance magnitude:

$$\lim_{D \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

All data points lie in a thin shell on the outer surface of the space. Every point is equidistant from every other point!
When all points are essentially the same distance away, **nearest neighbor sorting becomes meaningless random noise**.

---

## 2. Practical Mitigation Strategies

1. **Dimensionality Reduction (PCA):** Apply PCA to retain 95% variance and project 50+ features into 5-15 principal components.
2. **Feature Selection:** Use `SelectKBest` or drop low-variance noise features before passing to KNN.
3. **Use Manhattan Distance ($p=1$):** Lower $L_p$ norms suffer significantly less from the equidistance phenomenon than higher norms ($L_2$).
