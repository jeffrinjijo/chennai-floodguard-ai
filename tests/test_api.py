"""API tests - need the exported data (python scripts/05_export_api_data.py first)."""
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("xgboost")
from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["cells"] > 1000


def test_cells_geojson():
    g = client.get("/cells.geojson").json()
    assert g["type"] == "FeatureCollection" and len(g["features"]) > 1000


def test_predict_inside_study_area():
    r = client.get("/predict", params={"lat": 13.0827, "lon": 80.2707})
    if r.status_code == 404:                      # that exact spot may be sea; try a known land point
        first = client.get("/cells.geojson").json()["features"][0]["geometry"]["coordinates"][0][0]
        r = client.get("/predict", params={"lat": first[1] + 0.001, "lon": first[0] + 0.001})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["score"] <= 1 and 1 <= d["risk_class"] <= 5
    assert len(d["main_drivers"]) == 5 and "NOT a flood forecast" in d["disclaimer"]


def test_predict_outside_study_area_is_404():
    assert client.get("/predict", params={"lat": 28.6, "lon": 77.2}).status_code == 404


def test_predict_validates_input():
    assert client.get("/predict", params={"lat": 999, "lon": 80}).status_code == 422
