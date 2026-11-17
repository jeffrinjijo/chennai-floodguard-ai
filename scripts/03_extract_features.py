"""Step 4 - per-cell terrain + land-cover features, land mask, and a first signal check.

Inputs : data/interim/grid_labels.gpkg, data/raw/terrain/copernicus_dem_chennai.tif,
         data/raw/landcover/worldcover_2021_chennai.tif
Outputs: data/processed/features.csv / features.gpkg, reports/feature_summary.md

Features per 500 m cell
  elev_mean/min/std, slope_mean, rel_elev (cell minus ~3.5 km neighbourhood mean),
  frac_{tree,shrub,grass,crop,built,bare,water,wetland,mangrove},
  dist_sea_m, dist_inland_water_m (from WorldCover water pixels; thin rivers may be missed)
Mask: in_study = not mostly sea and not mostly water, with a valid DEM.

Run: python scripts/03_extract_features.py
"""
from __future__ import annotations
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs/config.yaml"))
CRS_M, CELL = CFG["geo"]["metric_crs"], CFG["geo"]["grid_cell_m"]
DEM_RES, WC_RES = 25, 10          # both divide CELL (500) exactly
WC_CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 80: "water", 90: "wetland", 95: "mangrove"}

grid = gpd.read_file(ROOT / "data/interim/grid_labels.gpkg").reset_index(drop=True)
b = grid.geometry.bounds
X0, Y0 = b.minx.min(), b.miny.min()
grid["ix"] = ((b.minx - X0) / CELL).round().astype(int)
grid["iy"] = ((b.miny - Y0) / CELL).round().astype(int)
NX, NY = int(grid.ix.max()) + 1, int(grid.iy.max()) + 1
assert len(grid) == NX * NY, "grid is not a full rectangle"


def to_metric(path: Path, res: int, resampling, dtype, nodata):
    """Reproject a raster onto the grid extent at `res` metres (rows run top->bottom)."""
    W, H = NX * CELL // res, NY * CELL // res
    tf = from_origin(X0, Y0 + NY * CELL, res, res)
    dst = np.full((H, W), nodata, dtype=dtype)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tf, dst_crs=CRS_M, resampling=resampling,
                  src_nodata=src.nodata, dst_nodata=nodata)
    return dst


def cellstat(arr: np.ndarray, res: int, fn) -> np.ndarray:
    """Aggregate a raster to the grid cells with fn(axis=(1,3)); returns one value per grid row."""
    k = CELL // res
    out = fn(arr.reshape(NY, k, NX, k), axis=(1, 3))[::-1, :]      # flip: row 0 -> southernmost
    return out[grid.iy.values, grid.ix.values]


# ---------------- terrain ----------------
dem = to_metric(ROOT / "data/raw/terrain/copernicus_dem_chennai.tif", DEM_RES,
                Resampling.bilinear, "float32", np.nan)
gy, gx = np.gradient(np.nan_to_num(dem, nan=0.0), DEM_RES)
slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
slope[np.isnan(dem)] = np.nan
grid["elev_mean"] = cellstat(dem, DEM_RES, np.nanmean)
grid["elev_min"] = cellstat(dem, DEM_RES, np.nanmin)
grid["elev_std"] = cellstat(dem, DEM_RES, np.nanstd)
grid["slope_mean"] = cellstat(slope, DEM_RES, np.nanmean)

# ---------------- land cover ----------------
wc = to_metric(ROOT / "data/raw/landcover/worldcover_2021_chennai.tif", WC_RES,
               Resampling.nearest, "uint8", 0)
for code, name in WC_CLASSES.items():
    grid[f"frac_{name}"] = cellstat((wc == code).astype("float32"), WC_RES, np.mean)

water = wc == 80
core = ndi.binary_opening(water, iterations=10)                  # drops channels < ~200 m wide
lab, n = ndi.label(core)
if n:
    sizes = ndi.sum(core, lab, range(1, n + 1))
    core = lab == (int(np.argmax(sizes)) + 1)                    # biggest body = the sea
sea = ndi.binary_dilation(core, iterations=10) & water
inland = water & ~sea
grid["sea_frac"] = cellstat(sea.astype("float32"), WC_RES, np.mean)
grid["dist_sea_m"] = cellstat(ndi.distance_transform_edt(~sea, sampling=WC_RES).astype("float32"), WC_RES, np.mean)
if inland.any():
    d = ndi.distance_transform_edt(~inland, sampling=WC_RES).astype("float32")
    grid["dist_inland_water_m"] = cellstat(d, WC_RES, np.mean)
else:
    grid["dist_inland_water_m"] = np.nan

grid["in_study"] = ((grid.sea_frac < 0.5) & (grid.frac_water < 0.5) & grid.elev_mean.notna()).astype(int)

# relative elevation vs ~3.5 km neighbourhood (land cells only)
E = np.full((NY, NX), np.nan)
land = grid.in_study.values == 1
E[grid.iy.values[land], grid.ix.values[land]] = grid.elev_mean.values[land]
local = ndi.uniform_filter(np.where(np.isnan(E), np.nanmean(E), E), size=7)
grid["rel_elev"] = E[grid.iy.values, grid.ix.values] - local[grid.iy.values, grid.ix.values]

# ---------------- outputs ----------------
feat_cols = ["elev_mean", "elev_min", "elev_std", "slope_mean", "rel_elev", "dist_sea_m",
             "dist_inland_water_m"] + [f"frac_{v}" for v in WC_CLASSES.values()]
out = ROOT / "data/processed"
out.mkdir(parents=True, exist_ok=True)
grid.drop(columns=["ix", "iy"]).to_file(out / "features.gpkg", driver="GPKG")
grid.drop(columns=["geometry", "ix", "iy"]).to_csv(out / "features.csv", index=False)

s = grid[grid.in_study == 1]
L = ["# Feature summary", "",
     f"Cells: {len(grid):,} total, **{len(s):,} kept** (sea / open-water cells removed: {len(grid) - len(s):,}).", "",
     "## Label rates in kept cells", "", "| label | positives | share |", "|---|---|---|"]
labels = ["pos_2005", "pos_2015", "pos_stag_2015", "pos_hot_2020"]
for c in labels:
    L.append(f"| {c} | {int(s[c].sum()):,} | {s[c].mean():.1%} |")

L += ["", "## First signal check: AUC of single features (0.5 = no signal)", "",
      "Higher AUC = the feature alone separates flooded from non-flooded cells. "
      "Elevation, slope and distance are flipped so that 'lower/closer = more flood' scores above 0.5.", "",
      "| feature | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
sign = {"elev_mean": -1, "elev_min": -1, "rel_elev": -1, "slope_mean": -1, "dist_sea_m": -1,
        "dist_inland_water_m": -1, "elev_std": 1}
for f in feat_cols:
    row = []
    for c in labels:
        d = s[[f, c]].dropna()
        ok = d[c].nunique() == 2
        row.append(f"{roc_auc_score(d[c], sign.get(f, 1) * d[f]):.2f}" if ok else "-")
    L.append(f"| {f} | " + " | ".join(row) + " |")

L += ["", "## Mean elevation (m): flooded vs not", "", "| label | flooded | not flooded |", "|---|---|---|"]
for c in labels:
    L.append(f"| {c} | {s.loc[s[c] == 1, 'elev_mean'].mean():.1f} | {s.loc[s[c] == 0, 'elev_mean'].mean():.1f} |")
L += ["", "## Notes", "",
      "- Copernicus DEM is a surface model (includes buildings/trees), not bare earth.",
      "- Distance to inland water comes from WorldCover pixels, not a river map; narrow or covered channels are missed.",
      "- No rainfall or drainage features yet."]
(ROOT / "reports/feature_summary.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
