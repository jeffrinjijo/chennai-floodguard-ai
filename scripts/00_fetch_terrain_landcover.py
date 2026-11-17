"""Fetch elevation + land cover for the Chennai study bbox WITHOUT any login.

* Elevation : Copernicus GLO-30 DEM (30 m, open data on AWS)
* Land cover: ESA WorldCover 2021 v200 (10 m, open data on AWS)

Both are cloud-optimised GeoTIFFs, so only the pixels inside the bbox are downloaded
(a few MB instead of hundreds). Run on your own machine (needs normal internet):

    python scripts/00_fetch_terrain_landcover.py
"""
from pathlib import Path
import sys

import rasterio
import yaml
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]
BBOX = yaml.safe_load(open(ROOT / "configs/config.yaml"))["geo"]["bbox"]
PAD = 0.02
BOUNDS = (BBOX[0] - PAD, BBOX[1] - PAD, BBOX[2] + PAD, BBOX[3] + PAD)

DEM = ("https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")
WC = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
      "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")

JOBS = {
    "terrain/copernicus_dem_chennai.tif": [
        DEM.format(lat="N12", lon="E080"), DEM.format(lat="N13", lon="E080")],
    "landcover/worldcover_2021_chennai.tif": [WC.format(tile="N12E078")],
}


def fetch(urls: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    srcs = [rasterio.open(u) for u in urls]
    try:
        arr, tf = merge(srcs, bounds=BOUNDS)
        meta = srcs[0].meta.copy()
        meta.update(height=arr.shape[1], width=arr.shape[2], transform=tf, compress="deflate")
        for k in ("blockxsize", "blockysize", "tiled"):
            meta.pop(k, None)
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(arr)
    finally:
        for s in srcs:
            s.close()
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {arr.shape[2]}x{arr.shape[1]} px)")


if __name__ == "__main__":
    ok = True
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        for rel, urls in JOBS.items():
            try:
                fetch(urls, ROOT / "data/raw" / rel)
            except Exception as e:
                ok = False
                print(f"FAILED {rel}: {e!r}\n  tried: {urls}")
    sys.exit(0 if ok else 1)
