"""FastAPI app: serves the map page and a small JSON API.

Run locally:  uvicorn backend.main:app --reload     then open http://127.0.0.1:8000
"""
from pathlib import Path

import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from backend import service

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"

app = FastAPI(title="Chennai FloodGuard AI", version="0.1.0",
              description=service.DISCLAIMER)

CELLS = service.load_cells(DATA / "cells.csv")
BOOSTER = xgb.Booster()
BOOSTER.load_model(str(DATA / "model.json"))
FEATS = list(BOOSTER.feature_names)
GEOJSON = service.build_geojson(CELLS)          # built once at startup


@app.get("/health")
def health():
    return {"status": "ok", "cells": len(CELLS), "features": FEATS}


@app.get("/cells.geojson")
def cells_geojson():
    return GEOJSON


@app.get("/predict")
def predict(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    row, dist = service.nearest_cell(CELLS, lat, lon)
    if row is None:
        raise HTTPException(404, f"Point is {dist:.0f} m from the nearest study cell - outside the study area "
                                 "(core Chennai land, 500 m cells).")
    return {
        "cell_id": int(row.cell_id), "lat": float(row.lat), "lon": float(row.lon),
        "score": round(float(row.score), 3), "risk_class": int(row.risk_class),
        "risk": service.RISK_LABELS[int(row.risk_class)],
        "flooded_in": service.history(row),
        "main_drivers": service.top_drivers(BOOSTER, row, FEATS),
        "drivers_note": "Contributions come from the final model; the score is the out-of-sample estimate.",
        "disclaimer": service.DISCLAIMER,
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend/index.html")


@app.get("/scenario")
def scenario(rain_mm_24h: float = Query(..., ge=0, le=1000)):
    c = service.rain_category(rain_mm_24h)
    return {"rain_mm_24h": rain_mm_24h, **c,
            "flagged_cells": service.count_flagged(CELLS, rain_mm_24h), "total_cells": len(CELLS),
            "method": service.SCREENING_METHOD, "disclaimer": service.DISCLAIMER}


@app.get("/rain-reference")
def rain_reference():
    return {"events": service.RAIN_REFERENCE, "note": service.RAIN_REFERENCE_NOTE}


@app.get("/outlook")
def outlook():
    """Next-3-day rainfall from a public forecast service, passed through the same screening rule."""
    try:
        data = service.fetch_outlook()
    except Exception as e:                                       # network down, service changed, bad payload
        raise HTTPException(status_code=503, detail=f"Rainfall outlook unavailable right now ({type(e).__name__}).")
    for d in data["days"]:
        d["flagged_cells"] = service.count_flagged(CELLS, d["rain_mm_24h"])
    return {**data, "total_cells": len(CELLS), "method": service.SCREENING_METHOD, "disclaimer": service.DISCLAIMER}


METRICS = service.load_metrics(DATA / "metrics.json")


@app.get("/dashboard")
def dashboard():
    return FileResponse(ROOT / "frontend/dashboard.html")


@app.get("/stats")
def stats():
    counts = CELLS["risk_class"].value_counts().sort_index()
    return {"cells": len(CELLS), "risk_class_counts": {int(k): int(v) for k, v in counts.items()},
            "metrics": METRICS, "disclaimer": service.DISCLAIMER}
