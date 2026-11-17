# Model report

Land cells: 1,930. Training/eval cells: **721** (281 recurrent, 440 never-flooded). Ambiguous (flooded in exactly 1 event): 1,209, scored but not used for training.
Spatial CV = 5 folds over 5 km blocks. Prevalence (chance PR-AUC) = 0.39.

## Headline: spatial cross-validation

| model | ROC-AUC | PR-AUC | P@k | Brier | ECE | fold AUC (mean ± sd) |
|---|---|---|---|---|---|---|
| Elevation only (LR) | 0.71 | 0.50 | 0.58 | 0.22 | 0.10 | 0.69 ± 0.11 |
| Logistic Regression | 0.81 | 0.71 | 0.66 | 0.18 | 0.06 | 0.80 ± 0.04 |
| Random Forest | 0.82 | 0.75 | 0.66 | 0.17 | 0.05 | 0.81 ± 0.06 |
| XGBoost | 0.81 | 0.75 | 0.66 | 0.18 | 0.08 | 0.80 ± 0.05 |

## Random CV vs spatial CV (leakage check)

| model | random ROC-AUC | spatial ROC-AUC |
|---|---|---|
| Elevation only (LR) | 0.71 | 0.71 |
| Logistic Regression | 0.83 | 0.81 |
| Random Forest | 0.86 | 0.82 |
| XGBoost | 0.87 | 0.81 |

A big gap means random CV is leaking neighbouring cells; trust the spatial numbers.

## Stress test: leave-one-event-out

Train on cells flooded in any OTHER event, test on the held-out event (spatial folds, so no neighbour leakage).

| held-out event | model | prevalence | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 2005 | Elevation only (LR) | 0.32 | 0.70 | 0.48 |
| 2005 | XGBoost | 0.32 | 0.34 | 0.23 |
| 2015 | Elevation only (LR) | 0.59 | 0.51 | 0.56 |
| 2015 | XGBoost | 0.59 | 0.37 | 0.53 |
| 2020 | Elevation only (LR) | 0.02 | 0.64 | 0.03 |
| 2020 | XGBoost | 0.02 | 0.58 | 0.03 |

## Mean |SHAP| (top 8)

- rel_elev: 0.742
- elev_mean: 0.710
- elev_min: 0.296
- elev_std: 0.283
- dist_sea_m: 0.263
- frac_wetland: 0.256
- frac_built: 0.242
- dist_inland_water_m: 0.199

## Read this before quoting numbers

- The score ranks cells by recurrent flood-proneness. It is NOT a flood probability and NOT a warning.
- Negatives mean 'not seen flooded' in three imperfect records, so some are really positives.
- ROC-AUC near 0.5 = no skill. Report spatial-CV numbers, and the leave-one-event-out table as honest context.
- Only terrain and land-cover features; no rainfall, drainage network or drain capacity.
- Cells were labelled from 3 events only; treat results as a prototype.