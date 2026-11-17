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


def test_scenario_levels():
    assert client.get("/scenario", params={"rain_mm_24h": 0}).json()["flagged_cells"] == 0
    d = client.get("/scenario", params={"rain_mm_24h": 285}).json()
    assert d["category"] == "extremely heavy" and d["flagged_classes"] == [3, 4, 5]
    assert d["flagged_cells"] > 0 and "Not a trained model" in d["method"]


def test_scenario_validates_input():
    assert client.get("/scenario", params={"rain_mm_24h": -5}).status_code == 422
    assert client.get("/scenario", params={"rain_mm_24h": 5000}).status_code == 422


def test_rain_reference():
    d = client.get("/rain-reference").json()
    assert len(d["events"]) == 4 and "approximate" in d["note"]


def test_outlook_ok_with_mocked_service(monkeypatch):
    from backend import service
    service._outlook_cache.update(t=0.0, data=None)
    mock = {"daily": {"time": ["2026-11-18", "2026-11-19", "2026-11-20"], "precipitation_sum": [2.0, 125.0, 40.0]}}
    monkeypatch.setattr(service, "_urlreq", type("U", (), {"urlopen": staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network in tests")))}))
    real = service.fetch_outlook
    monkeypatch.setattr(service, "fetch_outlook", lambda: real(http_get=lambda u: mock))
    r = client.get("/outlook")
    assert r.status_code == 200
    j = r.json()
    assert len(j["days"]) == 3 and j["peak"]["category"] == "very heavy"
    assert j["days"][1]["flagged_cells"] > 0 and j["days"][0]["flagged_cells"] == 0
    service._outlook_cache.update(t=0.0, data=None)


def test_outlook_503_when_service_down(monkeypatch):
    from backend import service
    def boom():
        raise OSError("network down")
    monkeypatch.setattr(service, "fetch_outlook", boom)
    r = client.get("/outlook")
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_stats_and_dashboard():
    r = client.get("/stats")
    assert r.status_code == 200
    j = r.json()
    assert j["cells"] > 1000 and sum(j["risk_class_counts"].values()) == j["cells"]
    assert 0.5 < j["metrics"]["roc_auc"]["xgb"] < 1.0         # run scripts/06_scientific_analysis.py first
    d = client.get("/dashboard")
    assert d.status_code == 200 and "FloodGuard" in d.text
