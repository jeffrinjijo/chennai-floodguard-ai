# Feature summary

Cells: 2,680 total, **1,930 kept** (sea / open-water cells removed: 750).

## Label rates in kept cells

| label | positives | share |
|---|---|---|
| pos_2005 | 612 | 31.7% |
| pos_2015 | 982 | 50.9% |
| pos_stag_2015 | 490 | 25.4% |
| pos_hot_2020 | 31 | 1.6% |

## First signal check: AUC of single features (0.5 = no signal)

Higher AUC = the feature alone separates flooded from non-flooded cells. Elevation, slope and distance are flipped so that 'lower/closer = more flood' scores above 0.5.

| feature | pos_2005 | pos_2015 | pos_stag_2015 | pos_hot_2020 |
|---|---|---|---|---|
| elev_mean | 0.70 | 0.50 | 0.47 | 0.62 |
| elev_min | 0.62 | 0.54 | 0.50 | 0.60 |
| elev_std | 0.35 | 0.52 | 0.47 | 0.37 |
| slope_mean | 0.71 | 0.37 | 0.41 | 0.53 |
| rel_elev | 0.68 | 0.53 | 0.44 | 0.50 |
| dist_sea_m | 0.47 | 0.59 | 0.58 | 0.57 |
| dist_inland_water_m | 0.56 | 0.43 | 0.44 | 0.56 |
| frac_tree | 0.55 | 0.46 | 0.40 | 0.32 |
| frac_shrub | 0.57 | 0.42 | 0.41 | 0.43 |
| frac_grass | 0.71 | 0.36 | 0.29 | 0.48 |
| frac_crop | 0.73 | 0.33 | 0.29 | 0.45 |
| frac_built | 0.36 | 0.62 | 0.73 | 0.70 |
| frac_bare | 0.64 | 0.43 | 0.46 | 0.56 |
| frac_water | 0.52 | 0.45 | 0.45 | 0.56 |
| frac_wetland | 0.54 | 0.43 | 0.44 | 0.47 |
| frac_mangrove | 0.50 | 0.50 | 0.50 | 0.50 |

## Mean elevation (m): flooded vs not

| label | flooded | not flooded |
|---|---|---|
| pos_2005 | 9.0 | 13.0 |
| pos_2015 | 11.4 | 12.1 |
| pos_stag_2015 | 11.7 | 11.7 |
| pos_hot_2020 | 9.2 | 11.7 |

## Notes

- Copernicus DEM is a surface model (includes buildings/trees), not bare earth.
- Distance to inland water comes from WorldCover pixels, not a river map; narrow or covered channels are missed.
- No rainfall or drainage features yet.