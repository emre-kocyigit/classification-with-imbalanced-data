# Classification with Imbalanced Data

> A practical, end-to-end guide for ML practitioners working with heavily skewed datasets.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Credit%20Card%20Fraud-orange)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Problem

Standard ML models fail silently on imbalanced data. A classifier that always predicts 
"legitimate" on a 99.8% / 0.2% fraud dataset scores 99.8% accuracy — and catches 
zero frauds. This notebook shows exactly what to do instead.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 
— Kaggle (`mlg-ulb/creditcardfraud`)

- 284,807 transactions · 492 fraud cases · **0.172% positive rate**
- Features V1–V28: PCA-transformed (anonymised)
- Recommended by Kaggle to evaluate using **PR-AUC**, not accuracy

## What's Covered

| Section | Topic |
|---------|-------|
| §2 | The accuracy paradox — toy dataset demo |
| §4 | EDA — class imbalance, Amount distribution, feature correlations |
| §7 | Baseline — Logistic Regression, no intervention |
| §8 | Class weights — one-parameter fix, zero data modification |
| §9 | Threshold tuning — free optimisation on any trained model |
| §10 | Resampling — SMOTE, ADASYN, SMOTETomek inside leakage-proof pipelines |
| §11 | Stronger models — Random Forest & XGBoost with class weighting |
| §12 | PR curve vs ROC curve — visual proof of why PR-AUC is the honest metric |
| §14 | Final comparison table across all methods |

## Key Results

| Method | Recall | PR-AUC |
|--------|--------|--------|
| Baseline LR | — | — |
| + Class Weights | — | — |
| + Threshold Tuning | — | — |
| Random Forest | — | — |
| XGBoost | — | — |

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/imbalanced-data-classification.git
cd imbalanced-data-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
#    Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place creditcard.csv in the root of this repo (it is gitignored)

# 4. Launch Jupyter
jupyter notebook imbalanced_data_classification.ipynb
```

## Core Lessons

TBD

## References

- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- Davis & Goadrich (2006) — *The Relationship Between Precision-Recall and ROC Curves*
