# Data download checklist

Raw data is **not committed** (see `.gitignore`). Download into the folders below.
Keep original file names. Total expected size: ~200-300 MB (under 50 MB after cropping).

## floods/  (all public domain, source: GCC flood DSS, https://www.chennaifloodsdss.in)
From https://data.opencity.in/dataset/chennai-flooding-data
- [ ] Chennai Inundation Points with Depth of Inundation (depth in inches)  <- best labels
- [ ] Chennai Flooding Points in 2015
- [ ] Chennai Flood Hazard Zones Map
- [ ] (optional) Flows for 5/10/25/50/100/200-year return periods

From https://data.opencity.in/dataset/chennai-floods-2015-data
- [ ] GCC Stagnation Locations in 2015
- [ ] Chennai 2015 Crowd-sourced Flooding Locations
- [ ] Chennai 2015 GCC Area Flood Hotspots (KML)
- [ ] Chennai 2015 Floods Inundation Zone (KML, ~5 MB)

Other events:
- [ ] Chennai GCC Flood Extent 2005 (KML)  - https://data.opencity.in/dataset/chennai-floods-2005-data
- [ ] Chennai GCC Flood Hotspots 2020 (KML) - https://data.opencity.in/dataset/chennai-floods-2020-data

## rainfall/
From https://data.opencity.in/dataset/chennai-rainfall-data
- [ ] Chennai Daily Rainfall Data from 1991-2023 (CSV, source TNDMA)
- [ ] (optional) Chennai Monthly Rainfall 1901-2021

## terrain/
- [ ] SRTM 1-arc-second tile covering Chennai (N13 E080) - USGS EarthExplorer (free login)

## landcover/
- [ ] ESA WorldCover tile covering Chennai - https://esa-worldcover.org/en
      (crop to Chennai afterwards, then delete the original)

## boundaries/
- [ ] Greater Chennai Corporation ward boundaries (search OpenCity / GCC)
- [ ] River and water-body lines/polygons (OpenStreetMap extract is fine for the demo)

## Skipped on purpose (too large for the reel version)
Sentinel-1 scenes, NASA IMERG, full Tamil Nadu rainfall files.

## Then run
```
python scripts/01_inventory.py
```
and open `reports/data_inventory.md`.
