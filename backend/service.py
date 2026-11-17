"""Pure logic for the API (no web framework, so it is easy to test)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

CELL_M = 500
M_PER_DEG_LAT = 111_320.0
MAX_SNAP_M = 600           # a click further than this from any cell centre is "outside the study area"
RISK_LABELS = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
EVENT_COLS = {"pos_2005": "2005", "e2015": "2015", "pos_hot_2020": "2020"}
DISCLAIMER = ("Experimental research demo built from public historical flood records. "
              "A relative ranking of recurrent flood-proneness - NOT a flood forecast, warning or probability.")


def load_cells(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["risk_label"] = df["risk_class"].map(RISK_LABELS)
    return df


def cell_polygon(lon: float, lat: float) -> list[list[list[float]]]:
    dlat = (CELL_M / 2) / M_PER_DEG_LAT
    dlon = (CELL_M / 2) / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]
    return [[[round(x, 6), round(y, 6)] for x, y in ring]]


def history(row) -> list[str]:
    return [yr for col, yr in EVENT_COLS.items() if int(row[col]) == 1]


def build_geojson(cells: pd.DataFrame) -> dict:
    feats = []
    for r in cells.itertuples(index=False):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": cell_polygon(r.lon, r.lat)},
            "properties": {"cell_id": int(r.cell_id), "score": round(float(r.score), 3),
                           "risk_class": int(r.risk_class), "risk": r.risk_label,
                           "flooded_in": history(r._asdict())},
        })
    return {"type": "FeatureCollection", "features": feats}


def distance_m(lat1, lon1, lat2, lon2):
    """Equirectangular distance in metres (fine at city scale); works on numpy arrays."""
    x = np.radians(lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    y = np.radians(lat2 - lat1)
    return 6_371_000.0 * np.hypot(x, y)


def nearest_cell(cells: pd.DataFrame, lat: float, lon: float):
    d = distance_m(lat, lon, cells["lat"].values, cells["lon"].values)
    i = int(np.argmin(d))
    return (cells.iloc[i], float(d[i])) if d[i] <= MAX_SNAP_M else (None, float(d[i]))


def top_drivers(booster, row: pd.Series, feats: list[str], k: int = 5) -> list[dict]:
    """Per-feature contributions to the margin from the final model (XGBoost's built-in SHAP values)."""
    import xgboost as xgb
    dm = xgb.DMatrix(row[feats].astype(float).to_frame().T, feature_names=feats)
    contrib = booster.predict(dm, pred_contribs=True)[0][:-1]       # last column is the bias term
    order = np.argsort(-np.abs(contrib))[:k]
    return [{"feature": feats[i], "value": round(float(row[feats[i]]), 3),
             "contribution": round(float(contrib[i]), 3),
             "effect": "raises score" if contrib[i] > 0 else "lowers score"} for i in order]


# ---------------- rainfall scenario screening ----------------
# IMD 24-hour rainfall categories (mm). This is a transparent screening RULE, not a trained model:
# the heavier the rain, the more susceptibility classes are flagged as "worth watching".
RAIN_LEVELS = [   # (lower bound mm in 24 h, category, flagged risk classes)
    (204.5, "extremely heavy", [3, 4, 5]),
    (115.6, "very heavy", [4, 5]),
    (64.5, "heavy", [5]),
    (0.0, "below heavy", []),
]

# Largest 3-day rainfall events in the station CSV (city-wide mean of the stations that reported).
RAIN_REFERENCE = [
    {"event": "Oct 2005 heavy rain", "date": "2005-10-28", "rain_72h_mm": 443, "rain_peak_day_mm": 238},
    {"event": "Michaung (Dec 2023)", "date": "2023-12-05", "rain_72h_mm": 365, "rain_peak_day_mm": 167},
    {"event": "Dec 2015 floods", "date": "2015-12-02", "rain_72h_mm": 342, "rain_peak_day_mm": 284},
    {"event": "Nivar (Nov 2020)", "date": "2020-11-26", "rain_72h_mm": 306, "rain_peak_day_mm": 136},
]
RAIN_REFERENCE_NOTE = ("City-wide mean over the 5-35 stations that reported on those days; early coverage is sparse, "
                       "so figures are approximate. Days missing from the source CSV were treated as no rain.")
SCREENING_METHOD = ("Rule of thumb: heavier 24 h rainfall flags more of the flood-prone classes. "
                    "Not a trained model and not a forecast. Check IMD and Greater Chennai Corporation for official warnings.")


def rain_category(mm: float) -> dict:
    for lo, name, classes in RAIN_LEVELS:
        if mm >= lo:
            return {"category": name, "flagged_classes": list(classes)}
    return {"category": "below heavy", "flagged_classes": []}


def count_flagged(cells: pd.DataFrame, mm: float) -> int:
    return int(cells["risk_class"].isin(rain_category(mm)["flagged_classes"]).sum())
