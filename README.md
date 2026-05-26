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

| Method | PR-AUC | ROC-AUC |
|--------|--------|---------|
| Baseline LR (no intervention) | 0.743 | 0.955 |
| LR + Class Weights | 0.720 | 0.971 |
| Random Forest (balanced weights) | 0.849 | 0.958 |
| XGBoost (scale_pos_weight) | **0.881** | 0.969 |
| Random classifier (baseline) | 0.0017 | 0.500 |

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

> Grounded in real results on the Kaggle Credit Card Fraud dataset (0.172% positive rate).

### 1. Accuracy is the wrong metric — always
A model that predicts "legitimate" for every transaction scores **99.83% accuracy** and catches
zero fraud. Never use accuracy to evaluate or compare models on imbalanced data.

### 2. PR-AUC is the honest metric. ROC-AUC flatters.
| | ROC-AUC range | PR-AUC range |
|-|---------------|--------------|
| All 4 models | 0.955 → 0.971 | 0.720 → 0.881 |

ROC-AUC suggests all models are roughly equivalent. PR-AUC reveals a 16-point gap.
ROC-AUC is diluted by the enormous TN count (284k legitimate transactions).
PR-AUC uses only TP, FP, FN — the majority class cannot inflate it.

### 3. Model choice matters more than resampling
SMOTE, ADASYN, and SMOTETomek all produced PR-AUC ≈ 0.75 with Logistic Regression.
XGBoost with just `scale_pos_weight` hit **PR-AUC = 0.881** — no resampling needed.
Pick the right model before reaching for synthetic data generation.

### 4. Class weights are your first line of defence
One parameter change. Zero data modification. No leakage risk.
Always try `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost) before
anything else. It is free and it should be your default.

### 5. Resampling shifts recall — it does not improve ranking
Oversampling and SMOTE dramatically increased Recall (0.63 → 0.95).
But PR-AUC barely moved. Resampling changes *where on the PR curve you operate*,
not the curve itself. Use it when you need high recall at a specific threshold,
not as a general-purpose fix.

### 6. Tune the threshold — it is free
The default 0.5 threshold assumes balanced classes. With 0.17% fraud it is almost
always wrong. Scanning the PR curve for the threshold that maximises F2 (recall-weighted)
costs nothing and can close most of the gap before any other intervention.

### 7. SMOTE inside the pipeline — or not at all
```python
# WRONG — leaks validation data into resampling
X_smote, y_smote = SMOTE().fit_resample(X_train, y_train)
cross_val_score(clf, X_smote, y_smote, cv=5)

# CORRECT — resampling stays inside each fold
pipe = ImbPipeline([('smote', SMOTE()), ('clf', clf)])
cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5))
```
Leakage inflates every metric and gives you false confidence in production.

### 8. The intervention hierarchy — cheapest first
```
1. Fix your metric        → PR-AUC, Recall, F-beta. Never accuracy.
2. Tune the threshold     → free, works on any trained model
3. Add class weights      → one parameter, should be your default
4. Use a stronger model   → RF / XGBoost beat LR + SMOTE on this dataset
5. Resample (SMOTE etc.)  → inside imblearn.Pipeline + StratifiedKFold only
6. Collect more data      → if none of the above is enough
7. Reframe the problem    → anomaly detection, one-class classification
```

## References

- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*
- Davis & Goadrich (2006) — *The Relationship Between Precision-Recall and ROC Curves*
