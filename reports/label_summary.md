# Label summary

Grid: **2,680** cells of 500 m (EPSG:32644), bbox [80.15, 12.9, 80.33, 13.2]. Positive = >= 10% of cell area inside flood extent.
Cells over the sea / outside the city are NOT yet masked - do that once the DEM is added.

| label | positive cells | share |
|---|---|---|
| pos_2005 | 615 | 22.9% |
| pos_2015 | 995 | 37.1% |
| pos_stag_2015 | 490 | 18.3% |
| pos_hot_2020 | 32 | 1.2% |

## Threshold sensitivity (extent events)

| min fraction | 2005 positives | 2015 positives |
|---|---|---|
| 0.001 | 767 | 1,423 |
| 0.05 | 677 | 1,152 |
| 0.1 | 615 | 995 |
| 0.25 | 483 | 700 |
| 0.5 | 322 | 285 |

## Cross-checks

- Cells flooded in both 2005 and 2015: **203**; only 2005: 412; only 2015: 792
- Of 490 cells with 2015 stagnation points, **68%** are also positive in the 2015 satellite extent (agreement between two independent 2015 label sources).
- Of 32 cells with 2020 hotspots, 28% were positive in 2015 and 62% in 2005.

## Caveats

- Extent polygons come from satellite passes, so water under cloud or at a different date may be missed.
- Point files (2015 stagnation, 2020 hotspots) are presence-only: absence of a point is not proof of no flood.
- 2020 has few points - use it as a small check set, not a full training event.