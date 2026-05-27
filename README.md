# Computational-Statistics-Project---BSC-Spring-2026
# Predicting House Sale Prices with Statistical Modeling and Neural Networks

**Computational Statistics — Spring 2026**

Authors: Gayakpa Kenny , Gatti Khaireddine

## Overview

This project analyzes the Kaggle *House Prices — Advanced Regression Techniques* dataset (1460 houses, 79 features) through a coherent progression of statistical and machine-learning methods, culminating in a Kaggle submission with a public leaderboard RMSE of **0.15591**.

The work is organized in five parts:

1. **Classical statistical inference** — descriptive statistics, confidence intervals, t-tests, Shapiro–Wilk normality
2. **ANOVA** — one-way and two-way for ordinal feature significance and interactions
3. **2³ factorial design** — main effects and interactions on three dominant factors
4. **Parametric regression** — OLS with Ridge/Lasso comparison (R² ≈ 0.82)
5. **Neural network** — multi-layer perceptron trained on all 79 features (CV RMSE 0.1821)

All analyses are run in parallel on three target scales (`SalePrice`, `SalePrice_Log`, `SalePrice_BoxCox`) to assess robustness across transformations.


- [Kaggle competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- [Course repository](https://github.com/lydiaYchen/CStat26/tree/main/Project)
