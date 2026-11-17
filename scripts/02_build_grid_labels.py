"""Step 3 - study grid + per-event flood labels.

Reads the flood KMLs in data/raw/floods (identified by layer name, not file name),
builds a regular grid over the study bbox in the metric CRS, and for every cell computes
  * flooded-area fraction for the 2005 and 2015 extent polygons
  * counts of 2015 stagnation points and 2020 hotspot points
Writes:
  data/interim/grid_labels.gpkg   (cells + labels)
  data/interim/grid_labels.csv
  reports/label_summary.md        (class balance, threshold sensitivity, label cross-checks)

Run:  python scripts/02_build_grid_labels.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import box

fiona.drvsupport.supported_drivers["KML"] = "rw"

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
CRS_M = CFG["geo"]["metric_crs"]
CELL = CFG["geo"]["grid_cell_m"]
BBOX = CFG["geo"]["bbox"]            # lon/lat: minx, miny, maxx, maxy
POS_FRAC = 0.10                      # cell is "flooded" if >=10% of its area is inside the extent

KEYS = {                              # substring of KML layer name -> role
    "irs_flood_extent_2005": "extent_2005",
    "nrsc_flood_extent_2015": "extent_2015",
    "stagnation_locations_2015": "points_2015",
    "hotspots_gcc_flood_hotspots_nem_2020": "points_2020",
}


def load_events() -> dict[str, gpd.GeoDataFrame]:
    found: dict[str, gpd.GeoDataFrame] = {}
    for p in sorted((ROOT / "data/raw/floods").glob("*.kml")):
        for layer in fiona.listlayers(p):
            for key, role in KEYS.items():
                if key in layer and role not in found:
                    g = gpd.read_file(p, driver="KML", layer=layer)
                    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
                    g["geometry"] = g.geometry.make_valid()
                    found[role] = g.to_crs(CRS_M)
                    print(f"loaded {role}: {len(g)} features from {p.name}")
    missing = set(KEYS.values()) - set(found)
    if missing:
        sys.exit(f"Missing flood layers: {sorted(missing)} - check data/raw/floods")
    return found


def make_grid() -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gpd.GeoSeries([box(*BBOX)], crs=4326).to_crs(CRS_M).total_bounds
    xs = np.arange(np.floor(minx / CELL) * CELL, maxx, CELL)
    ys = np.arange(np.floor(miny / CELL) * CELL, maxy, CELL)
    cells = [box(x, y, x + CELL, y + CELL) for x in xs for y in ys]
    g = gpd.GeoDataFrame({"cell_id": np.arange(len(cells))}, geometry=cells, crs=CRS_M)
    c = g.geometry.centroid.to_crs(4326)
    g["lon"], g["lat"] = c.x.round(6), c.y.round(6)
    return g


def area_fraction(grid: gpd.GeoDataFrame, flood: gpd.GeoDataFrame) -> pd.Series:
    clip = flood.clip(grid.total_bounds)
    union = clip.geometry.union_all()
    return grid.geometry.intersection(union).area / (CELL * CELL)


def point_counts(grid: gpd.GeoDataFrame, pts: gpd.GeoDataFrame) -> pd.Series:
    pts = pts.copy()
    pts["geometry"] = pts.geometry.force_2d()
    j = gpd.sjoin(pts[["geometry"]], grid[["cell_id", "geometry"]], predicate="within")
    return grid["cell_id"].map(j.groupby("cell_id").size()).fillna(0).astype(int)


def main() -> None:
    ev = load_events()
    grid = make_grid()
    print(f"grid: {len(grid)} cells of {CELL} m")

    grid["frac_2005"] = area_fraction(grid, ev["extent_2005"]).round(4)
    grid["frac_2015"] = area_fraction(grid, ev["extent_2015"]).round(4)
    grid["n_stag_2015"] = point_counts(grid, ev["points_2015"])
    grid["n_hot_2020"] = point_counts(grid, ev["points_2020"])
    grid["pos_2005"] = (grid.frac_2005 >= POS_FRAC).astype(int)
    grid["pos_2015"] = (grid.frac_2015 >= POS_FRAC).astype(int)
    grid["pos_stag_2015"] = (grid.n_stag_2015 > 0).astype(int)
    grid["pos_hot_2020"] = (grid.n_hot_2020 > 0).astype(int)

    out = ROOT / "data/interim"
    out.mkdir(parents=True, exist_ok=True)
    grid.to_file(out / "grid_labels.gpkg", driver="GPKG")
    grid.drop(columns="geometry").to_csv(out / "grid_labels.csv", index=False)

    n = len(grid)
    L = ["# Label summary", "",
         f"Grid: **{n:,}** cells of {CELL} m ({CRS_M}), bbox {BBOX}. Positive = >= {POS_FRAC:.0%} of cell area inside flood extent.",
         "Cells over the sea / outside the city are NOT yet masked - do that once the DEM is added.", "",
         "| label | positive cells | share |", "|---|---|---|"]
    for c in ["pos_2005", "pos_2015", "pos_stag_2015", "pos_hot_2020"]:
        L.append(f"| {c} | {int(grid[c].sum()):,} | {grid[c].mean():.1%} |")

    L += ["", "## Threshold sensitivity (extent events)", "", "| min fraction | 2005 positives | 2015 positives |", "|---|---|---|"]
    for t in [0.001, 0.05, 0.10, 0.25, 0.50]:
        L.append(f"| {t} | {int((grid.frac_2005 >= t).sum()):,} | {int((grid.frac_2015 >= t).sum()):,} |")

    both = int(((grid.pos_2005 == 1) & (grid.pos_2015 == 1)).sum())
    only05 = int(((grid.pos_2005 == 1) & (grid.pos_2015 == 0)).sum())
    only15 = int(((grid.pos_2005 == 0) & (grid.pos_2015 == 1)).sum())
    L += ["", "## Cross-checks", "",
          f"- Cells flooded in both 2005 and 2015: **{both:,}**; only 2005: {only05:,}; only 2015: {only15:,}"]
    sp = grid[grid.pos_stag_2015 == 1]
    if len(sp):
        L.append(f"- Of {len(sp):,} cells with 2015 stagnation points, **{sp.pos_2015.mean():.0%}** are also positive in the 2015 satellite extent "
                 "(agreement between two independent 2015 label sources).")
    hp = grid[grid.pos_hot_2020 == 1]
    if len(hp):
        L.append(f"- Of {len(hp):,} cells with 2020 hotspots, {hp.pos_2015.mean():.0%} were positive in 2015 and {hp.pos_2005.mean():.0%} in 2005.")
    L += ["", "## Caveats", "",
          "- Extent polygons come from satellite passes, so water under cloud or at a different date may be missed.",
          "- Point files (2015 stagnation, 2020 hotspots) are presence-only: absence of a point is not proof of no flood.",
          "- 2020 has few points - use it as a small check set, not a full training event."]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/label_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
