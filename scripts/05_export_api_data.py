"""Step 6 - export the small files the API needs (so the deployed app is self-contained).

Reads  data/processed/predictions.csv and models/xgb_susceptibility.json
Writes backend/data/cells.csv  and backend/data/model.json
Run: python scripts/05_export_api_data.py
"""
from pathlib import Path
import shutil

import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "backend/data"
out.mkdir(parents=True, exist_ok=True)

booster = xgb.Booster()
booster.load_model(str(ROOT / "models/xgb_susceptibility.json"))
feats = list(booster.feature_names)

df = pd.read_csv(ROOT / "data/processed/predictions.csv")
keep = ["cell_id", "lon", "lat", "score", "risk_class", "recurrent_label",
        "pos_2005", "e2015", "pos_hot_2020", "n_events"] + feats
df[keep].round(5).to_csv(out / "cells.csv", index=False)
shutil.copy(ROOT / "models/xgb_susceptibility.json", out / "model.json")
print(f"exported {len(df)} cells, {len(feats)} features -> {out}")
