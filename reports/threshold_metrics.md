# Classification metrics at a fixed threshold

Labelled cells: 721 (281 recurrent, 440 never recorded). 5-fold spatial block CV on 5 km blocks. Accuracy of always predicting the majority class: **0.61**. ROC-AUC is threshold-free and is repeated for reference.

## A. Decision rule: predict recurrent when score >= 0.5

| model | accuracy | balanced acc. | precision | recall | F1 | MCC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Elevation only (LR) | 0.65 | 0.66 | 0.54 | 0.68 | 0.60 | 0.31 | 0.71 |
| Logistic Regression | 0.72 | 0.73 | 0.62 | 0.75 | 0.68 | 0.45 | 0.81 |
| Random Forest | 0.73 | 0.72 | 0.65 | 0.67 | 0.66 | 0.44 | 0.82 |
| XGBoost | 0.74 | 0.73 | 0.65 | 0.69 | 0.67 | 0.46 | 0.81 |

Confusion matrices (rows = truth; counts of cells):

| model | TN | FP | FN | TP |
|---|---|---|---|---|
| Elevation only (LR) | 280 | 160 | 90 | 191 |
| Logistic Regression | 312 | 128 | 71 | 210 |
| Random Forest | 338 | 102 | 92 | 189 |
| XGBoost | 337 | 103 | 86 | 195 |

## B. Screening rule: flag the top 20% of cells by score

Flags 144 of 721 labelled cells.

| model | accuracy | precision | recall | F1 | MCC |
|---|---|---|---|---|---|
| Elevation only (LR) | 0.63 | 0.56 | 0.28 | 0.38 | 0.17 |
| Logistic Regression | 0.69 | 0.71 | 0.36 | 0.48 | 0.33 |
| Random Forest | 0.73 | 0.81 | 0.41 | 0.55 | 0.43 |
| XGBoost | 0.73 | 0.79 | 0.41 | 0.54 | 0.41 |

## Reading these numbers

- Accuracy alone is misleading on imbalanced classes; compare it with the majority-class baseline and read balanced accuracy, F1 and MCC next to it.
- The threshold is fixed in advance (0.5, or a fixed top-20% budget); nothing was tuned on the evaluation folds.
- Labels are noisy (some never-recorded cells really flooded), so precision is probably understated and these figures are conservative for the true task.
- Spatial-CV numbers describe skill on unseen neighbourhoods of the same three events, not on a new storm (see the leave-one-event-out table in model_report.md).