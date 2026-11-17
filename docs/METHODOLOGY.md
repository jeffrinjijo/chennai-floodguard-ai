# Methodology

All numbers quoted here are produced by the scripts in `scripts/` and written to `reports/`. Nothing is hand-edited.

## 1. Problem definition
**Task.** Rank 500 m land cells of core Chennai by *recurrent flood-proneness*: how consistently an area appears in historical flood records.
**Not the task.** Predicting whether a particular storm will flood a particular street. That needs rainfall, drainage capacity and event-level labels, none of which are in the training data.

## 2. Study area and grid
- Bounding box (lon/lat): 80.15, 12.90, 80.33, 13.20. Projected to EPSG:32644 (UTM 44N) so cell sizes and distances are in metres.
- Regular 500 m grid: 2,680 cells; 1,930 kept after removing cells that are mostly sea or open water (750 removed).
- Sea is separated from inland water by morphological opening of ESA WorldCover water pixels (removes channels narrower than about 200 m) and keeping the largest connected body.

## 3. Data
| Layer | Source | Resolution | Use |
|---|---|---|---|
| Flood extent 2005 | IRS satellite polygons, GCC flood DSS via OpenCity | vector | label |
| Flood extent 2015 | NRSC satellite polygons, GCC flood DSS | vector | label |
| Stagnation points 2015 | GCC | points | label (merged with the 2015 extent) |
| Flood hotspots 2020 | GCC | 53 points | label (small check set) |
| Elevation | Copernicus DEM GLO-30 | 30 m, reprojected to 25 m | features |
| Land cover | ESA WorldCover 2021 v200 | 10 m | features |

## 4. Label construction
Per cell and event, a cell is flooded if at least 10% of its area lies inside the event polygon (2005, 2015 extent) or if it contains at least one reported point (2015 stagnation, 2020 hotspot). The 2015 extent and stagnation points count as one event. `n_events` is the number of events (0 to 3) in which the cell was flooded.

| class | rule | cells | role |
|---|---|---|---|
| recurrent (positive) | n_events >= 2 | 281 | train and evaluate |
| never recorded (negative) | n_events = 0 | 440 | train and evaluate |
| ambiguous | n_events = 1 | 1,209 | excluded from training, still scored |

Why this design: single-event footprints disagree. The 2005 and 2015 footprints are negatively correlated (phi -0.22, Jaccard 0.18) because they come from different sensors, dates and definitions. Requiring two events removes event-specific noise and targets what the three records agree on. The cost is a smaller sample and a class (never recorded) that is not a true negative.

## 5. Features (15)
- **Terrain** (from the DEM, aggregated to each cell): mean, minimum and standard deviation of elevation; mean slope; `rel_elev` = cell elevation minus the mean in a ~3.5 km neighbourhood (a topographic-position index: low values are basins relative to their surroundings).
- **Proximity:** `dist_sea_m` and `dist_inland_water_m`, Euclidean distance transforms from WorldCover water pixels.
- **Land cover:** area fraction per WorldCover class present (tree, shrub, grass, crop, built-up, bare, water, wetland).
- Label columns (`frac_2005`, `frac_2015`, point counts) are never features. An explicit feature list and an assertion enforce this; an earlier version leaked them and scored an implausible AUC of 0.997, which is how the bug was caught.

## 6. Models
| Model | Configuration |
|---|---|
| Elevation-only logistic regression | one feature (`elev_mean`); the baseline any extra complexity must beat |
| Logistic regression | median imputation, standardisation, balanced class weights |
| Random forest | 400 trees, max depth 8, min leaf 5, balanced subsample weights |
| XGBoost | 300 trees, learning rate 0.03, max depth 3, min child weight 3, subsample 0.8, column subsample 0.8, `scale_pos_weight` from class ratio |

Class weighting means scores are **not calibrated probabilities**. They are used as a ranking, and risk classes are score-rank quintiles.

## 7. Validation design
1. **Spatial block cross-validation (headline).** Cells are grouped into 5 km x 5 km blocks (25 blocks with labelled cells); whole blocks are held out together (5 folds, `GroupKFold`). This prevents a model from being scored on cells whose neighbours it trained on.
2. **Random 5-fold CV** is reported only to quantify leakage (XGBoost 0.87 random vs 0.81 spatial ROC-AUC).
3. **Block bootstrap.** Confidence intervals resample blocks, not cells, with 2,000 draws (seed 42), so the interval reflects the real number of independent units.
4. **Leave-one-event-out stress test.** Train on cells flooded in any other event, test on the held-out event. This asks whether a model learns flood-proneness or just memorises one event's footprint.
5. **Sanity checks on unseen classes.** Single-event cells (never trained on) should score between the two labelled classes.
6. **Residual spatial autocorrelation** (Moran's I with 8 nearest neighbours and 999 permutations) shows what spatial structure the model leaves unexplained.

Metrics: ROC-AUC, PR-AUC (compared with prevalence 0.39), precision at k, Brier score, expected calibration error, and precision/recall/lift at fixed inspection budgets.

## 8. Interpretability
Per-cell drivers are XGBoost's exact tree-SHAP contributions (`pred_contribs`), served by `/predict`. They describe the model's behaviour, not physical causation.

## 9. Rainfall scenario screening (not a model)
The scenario slider applies a fixed rule on IMD 24 h rainfall categories (heavy 64.5, very heavy 115.6, extremely heavy 204.5 mm): heavier rain flags more of the susceptibility classes (very high only; high and very high; moderate to very high). The thresholds are a transparent assumption. There is no rainfall-to-flood training because the event labels are too few and too inconsistent to learn one. The reference events (2005, 2015, 2020, 2023) are shown for context only.

## 10. Reproducibility
Seed 42 throughout; dependencies listed in `requirements.txt`; the full pipeline is the script sequence in the README; tests (`pytest -q`) cover the scenario rule boundaries, the service logic and every API endpoint.

## 11. Threats to validity
- **Label noise:** satellite extents miss water under cloud and in dense urban areas; point records are presence-only. "Never recorded" is not "never flooded".
- **Small effective sample:** 25 spatial blocks; fold-to-fold ROC-AUC spread is about 0.05.
- **Event-specific footprints:** cross-event transfer is at or below chance for the multi-feature models (see `reports/model_report.md`).
- **Missing physics:** no rainfall intensity, drainage network, drain capacity, tidal level or river discharge.
- **DEM is a surface model** (includes buildings and canopy) at 25 to 30 m; `elev_min` includes zero-elevation sea pixels in coastal cells and partly acts as a coastline flag.
- **Scale:** 500 m cells describe neighbourhoods, not streets.
