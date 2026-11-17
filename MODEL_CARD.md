# Model card: Chennai FloodGuard susceptibility model

| | |
|---|---|
| Model | XGBoost binary classifier (300 trees, depth 3), scored on 500 m cells |
| Output | Relative score in [0, 1] and five risk classes (score-rank quintiles). Not a probability |
| Intended use | Educational and exploratory screening: showing which Chennai neighbourhoods appear repeatedly in past flood records, and what terrain and land-cover factors the model associates with that |
| Out of scope | Flood warnings, evacuation or insurance decisions, street-level or property-level risk, any use as a forecast. Official alerts come from IMD and the Greater Chennai Corporation |
| Training data | 721 labelled cells (281 recurrent, 440 never recorded) from 2005, 2015 and 2020 flood records; 15 terrain and land-cover features |
| Not trained on | 1,209 cells flooded in exactly one event (scored, not used for fitting) |

## Performance (spatial block CV, 25 blocks of 5 km)
| Metric | Value |
|---|---|
| ROC-AUC | 0.81, 95% block-bootstrap interval 0.76 to 0.86 |
| PR-AUC | 0.75 (interval 0.62 to 0.85); chance level 0.39 |
| Elevation-only baseline ROC-AUC | 0.71 (interval 0.60 to 0.80) |
| Gain over baseline | +0.10 ROC-AUC (interval +0.03 to +0.20) |
| Observed recurrent-flood rate, class 1 to 5 | 0.09, 0.31, 0.41, 0.50, 0.79 (prevalence 0.39) |
| Residual spatial autocorrelation | Moran's I 0.33 (p = 0.001): spatially structured error remains |

Full tables: `reports/model_report.md`, `reports/scientific_analysis.md`.

## Known failure modes
- **Does not transfer across events.** Trained on other events, the multi-feature models score below chance on 2005 (ROC-AUC 0.34) and 2015 (0.37). Treat the score as a description of the pooled historical record.
- **Central Chennai is under-ranked as *recurrent*.** It flooded heavily in 2015 but appears once, so it scores lower than cells flooded in several records.
- **Coastline artefact.** `elev_min` partly encodes proximity to the sea because coastal cells contain zero-elevation pixels.
- **Wetland and vegetation effects are probably labelling artefacts.** Satellite flood maps exclude permanent water and vegetated areas.
- **Class weighting** makes raw scores uncalibrated; do not read 0.7 as 70% chance.

## Ethical and safety considerations
A susceptibility map can be misread as a warning or as a property-value signal. The app shows a disclaimer on every view, the API returns it in each response, and the README states the scope. Residents should rely on official alerts.

## Data licences
Flood records: public domain (OpenCity / GCC). Copernicus DEM: contains modified Copernicus data. ESA WorldCover 2021: CC BY 4.0. Basemap: OpenStreetMap contributors.
