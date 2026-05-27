# Computational-Statistics-Project---BSC-Spring-2026
# Predicting House Sale Prices with Statistical Modeling and Neural Networks

**Computational Statistics — Spring 2026**

Authors: Gayakpa Kenny (22510580), Khaireddine Gatti

## Overview

This project analyzes the Kaggle *House Prices — Advanced Regression Techniques* dataset (1460 houses, 79 features) through a coherent progression of statistical and machine-learning methods, culminating in a Kaggle submission with a public leaderboard RMSE of **0.15591**.

The work is organized in five parts:

1. **Classical statistical inference** — descriptive statistics, confidence intervals, t-tests, Shapiro–Wilk normality
2. **ANOVA** — one-way and two-way for ordinal feature significance and interactions
3. **2³ factorial design** — main effects and interactions on three dominant factors
4. **Parametric regression** — OLS with Ridge/Lasso comparison (R² ≈ 0.82)
5. **Neural network** — multi-layer perceptron trained on all 79 features (CV RMSE 0.1821)

All analyses are run in parallel on three target scales (`SalePrice`, `SalePrice_Log`, `SalePrice_BoxCox`) to assess robustness across transformations.

## Repository Structure

```
.
├── README.md                                    # This file
├── Project  report Predicting_House_Sale_Prices.md          # Full project report
└── figures/                                     # All figures referenced in the report
    ├── saleprice_distribution.png
    ├── transformations_comparison.png
    ├── six_panel_distribution.png
    ├── crosstab_overview.png
    ├── crosstab_ovqual_extqual.png
    ├── crosstab_ovqual_bsmqual.png
    ├── crosstab_ovqual_centair.png
    ├── crosstab_extqual_kitqual.png
    ├── crosstab_centair_lotshp.png
    ├── post_merge_heatmaps.png
    ├── interaction_plots_saleprice.png
    ├── interaction_plots_log.png
    ├── interaction_plots_boxcox.png
    ├── effect_bars_saleprice.png
    ├── effect_bars_log.png
    ├── effect_bars_boxcox.png
    ├── residual_diagnostics_saleprice.png
    ├── residual_diagnostics_log.png
    ├── residual_diagnostics_boxcox.png
    └── mlp_residual_diagnostics.png
```


- [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- [Course repository](https://github.com/lydiaYchen/CStat26/tree/main/Project)
