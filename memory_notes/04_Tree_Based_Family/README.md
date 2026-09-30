# 🌳 Tree-Based Family Architecture & Memory Notes

Tree-based algorithms recursively partition feature space into orthogonal hyper-rectangles, delivering scale-invariant non-linear modeling with zero feature normalization requirements.

| Algorithm | Type | Core Splitting Rule | Subspace | Guide Index |
| :--- | :--- | :--- | :--- | :--- |
| **Decision Tree Classifier** | Single Tree | Gini Impurity / Entropy | Full ($p$) | [01_Decision_Tree_Classifier/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/04_Tree_Based_Family/01_Decision_Tree_Classifier/README.md) |
| **Decision Tree Regressor** | Single Tree | Variance Reduction / MSE | Full ($p$) | [02_Decision_Tree_Regressor/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/04_Tree_Based_Family/02_Decision_Tree_Regressor/README.md) |
| **Random Forest Classifier** | Bagging Ensemble | Gini / Entropy | $m = \sqrt{p}$ | [03_Random_Forest_Classifier/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/04_Tree_Based_Family/03_Random_Forest_Classifier/README.md) |
| **Random Forest Regressor** | Bagging Ensemble | Variance Reduction / MSE | $m = p / 3$ | [04_Random_Forest_Regressor/README.md](file:///home/python/03_Custom_addons/ML/memory_notes/04_Tree_Based_Family/04_Random_Forest_Regressor/README.md) |
