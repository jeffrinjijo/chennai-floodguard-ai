"""Step 2 - Data inventory.

Walks data/raw and writes reports/data_inventory.md describing every file:
size, format, and (where libraries are available) columns, CRS, row count,
date range, raster shape/resolution.

Usage:
    python scripts/01_inventory.py [--raw data/raw] [--out reports/data_inventory.md]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None
try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None

TABULAR = {".csv", ".tsv"}
VECTOR = {".kml", ".geojson", ".json", ".gpkg", ".shp"}
RASTER = {".tif", ".tiff", ".hgt"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def describe_tabular(p: Path) -> list[str]:
    if pd is None:
        return ["pandas not installed - skipped"]
    sep = "\t" if p.suffix == ".tsv" else ","
    df = pd.read_csv(p, sep=sep, low_memory=False)
    out = [f"rows: {len(df):,}  cols: {len(df.columns)}",
           f"columns: {', '.join(map(str, df.columns))}",
           f"missing cells: {int(df.isna().sum().sum()):,}",
           f"duplicate rows: {int(df.duplicated().sum()):,}"]
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower():
            d = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if d.notna().any():
                out.append(f"date range ({c}): {d.min().date()} -> {d.max().date()}"
                           f"  (unparseable: {int(d.isna().sum())})")
    return out


def describe_vector(p: Path) -> list[str]:
    if gpd is None:
        return ["geopandas not installed - skipped"]
    out = []
    if p.suffix == ".kml":
        import fiona  # noqa: WPS433
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
        layers = fiona.listlayers(p)
        out.append(f"layers: {layers}")
        for layer in layers:
            g = gpd.read_file(p, driver="KML", layer=layer)
            out += _gdf_summary(g, f"layer '{layer}'")
        return out
    return _gdf_summary(gpd.read_file(p), "")


def _gdf_summary(g, tag: str) -> list[str]:
    pre = f"[{tag}] " if tag else ""
    geom = ", ".join(sorted(set(g.geometry.geom_type.dropna())))
    b = g.total_bounds.round(4).tolist() if len(g) else None
    return [f"{pre}features: {len(g):,}  geometry: {geom}  CRS: {g.crs}",
            f"{pre}bounds (minx,miny,maxx,maxy): {b}",
            f"{pre}attributes: {', '.join(c for c in g.columns if c != 'geometry')}",
            f"{pre}empty/invalid geometries: "
            f"{int(g.geometry.is_empty.sum())}/{int((~g.geometry.is_valid).sum())}"]


def describe_raster(p: Path) -> list[str]:
    if rasterio is None:
        return ["rasterio not installed - skipped"]
    with rasterio.open(p) as r:
        return [f"size: {r.width}x{r.height}  bands: {r.count}  dtype: {r.dtypes[0]}",
                f"CRS: {r.crs}  resolution: {r.res}",
                f"bounds: {tuple(round(v, 4) for v in r.bounds)}  nodata: {r.nodata}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="reports/data_inventory.md")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    files = sorted(p for p in raw.rglob("*") if p.is_file() and p.name != ".gitkeep")
    lines = ["# Data inventory", "", f"Files found under `{raw}`: **{len(files)}**", ""]
    if not files:
        lines.append("_No data yet. See data/README.md for the download checklist._")
    for p in files:
        lines.append(f"## {p.relative_to(raw)}  ({human(p.stat().st_size)})")
        try:
            ext = p.suffix.lower()
            if ext in TABULAR:
                info = describe_tabular(p)
            elif ext in VECTOR:
                info = describe_vector(p)
            elif ext in RASTER:
                info = describe_raster(p)
            else:
                info = ["unrecognised format - inspect manually"]
        except Exception as e:  # keep going; the report should list failures
            info = [f"ERROR reading file: {e!r}"]
        lines += [f"- {i}" for i in info] + [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
