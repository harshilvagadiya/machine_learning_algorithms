# 06. Tree Visualization & Decision Surface Geometry

Welcome to Guide 06 of the **Decision Tree Architecture Suite**.

> **CRITICAL NOTE ON COMMON STEPS:**
> Standard pre-flight steps (Data ingestion, missing value imputation, EDA, train/test splitting, and ColumnTransformer preprocessing) are standardized in the [Universal Pre-Flight Architecture Guide](../../00_Common_ML_Workflow/README.md).
>
> This guide documents **visualizing decision trees in production, text export formats, and the orthogonal diagonal boundary trap**.

---

## 1. Visualizing the Tree: `plot_tree`

Scikit-Learn provides `plot_tree` to render full branching logic with color-coded node purity:

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 9), dpi=300)
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,             # Colors nodes by dominant class
    rounded=True,            # Rounded rectangular boxes
    fontsize=10,
    max_depth=3              # Limit visual depth for executive clarity
)
plt.title("Executive Decision Tree Governance View", fontweight='bold')
plt.show()
```

---

## 2. Text Export: `export_text` for Auditing & Governance

For compliance audits (GDPR, Fair Lending regulations), graphical images are difficult to store in database logs. 

`export_text` outputs exact ASCII logic rules:

```python
from sklearn.tree import export_text

tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)
```

```text
|--- OverTime_Yes <= 0.50
|   |--- MonthlyIncome <= 3500.00
|   |   |--- class: 0
|   |--- MonthlyIncome >  3500.00
|   |   |--- class: 0
|--- OverTime_Yes >  0.50
|   |--- JobSatisfaction <= 2.50
|   |   |--- class: 1 (High Attrition Risk)
|   |--- JobSatisfaction >  2.50
|   |   |--- class: 0
```

---

## 3. The Diagonal Boundary Trap (Geometric Failure Mode)

Because tree nodes evaluate single features ($x_j \le \theta$), decision boundaries can **only form horizontal and vertical step lines**.

```
       TRUE BOUNDARY: x1 + x2 = 1.0 (DIAGONAL)           TREE STEP-WISE APPROXIMATION
       
       x2 ▲                                              x2 ▲
          │      ●                                          │      ●
          │       ●                                         │    ┌─●
          │        ●  (Class 1)                             │    │  ●
          │         ●                                       │  ┌─┘   ●
          │          ●                                      │  │      ●
          │  ○        ●                                     │  │○      ●
          │   ○        ●                                    │┌─┘○       ●
          │    ○        ● (Class 0)                         ││   ○       ●
          └─────────────────────► x1                        └─────────────────────► x1
            Smooth Linear Separation                          Staircase Overfitting!
```

* To approximate a simple $45^\circ$ diagonal line, a tree must construct a **deep, convoluted staircase of $20+$ orthogonal cuts**.
* **Remedy:** If features have strong linear diagonal relationships, use Linear/Logistic models or apply PCA to rotate coordinates before tree fitting!
