import numpy as np
import pandas as pd
import pytest

from backend import service


@pytest.fixture
def cells():
    df = pd.DataFrame({"cell_id": [0, 1, 2], "lon": [80.20, 80.21, 80.30], "lat": [13.00, 13.00, 13.10],
                       "score": [0.9, 0.2, 0.5], "risk_class": [5, 1, 3], "recurrent_label": [1, 0, -1],
                       "pos_2005": [1, 0, 0], "e2015": [1, 0, 1], "pos_hot_2020": [0, 0, 0]})
    df["risk_label"] = df.risk_class.map(service.RISK_LABELS)
    return df


def test_polygon_is_closed_and_about_500m(cells):
    ring = service.cell_polygon(80.2, 13.0)[0]
    assert ring[0] == ring[-1] and len(ring) == 5
    width = service.distance_m(13.0, ring[0][0], 13.0, ring[1][0])
    assert width == pytest.approx(500, abs=5)


def test_nearest_cell_snaps_and_rejects_far_points(cells):
    row, d = service.nearest_cell(cells, 13.0005, 80.2002)
    assert row.cell_id == 0 and d < 100
    row, d = service.nearest_cell(cells, 12.0, 79.0)
    assert row is None and d > service.MAX_SNAP_M


def test_geojson_has_one_feature_per_cell_and_history(cells):
    g = service.build_geojson(cells)
    assert len(g["features"]) == 3
    assert g["features"][0]["properties"]["flooded_in"] == ["2005", "2015"]
    assert g["features"][1]["properties"]["flooded_in"] == []


def test_scores_are_between_0_and_1(cells):
    assert cells.score.between(0, 1).all()


def test_rain_category_boundaries():
    assert service.rain_category(0)["flagged_classes"] == []
    assert service.rain_category(64.4)["category"] == "below heavy"
    assert service.rain_category(64.5) == {"category": "heavy", "flagged_classes": [5]}
    assert service.rain_category(115.6)["flagged_classes"] == [4, 5]
    assert service.rain_category(285)["flagged_classes"] == [3, 4, 5]


def test_flagged_count_grows_with_rain(cells):
    counts = [service.count_flagged(cells, mm) for mm in (0, 70, 120, 250)]
    assert counts == sorted(counts) and counts[0] == 0 and counts[-1] >= counts[1]


def test_rain_reference_has_the_four_events():
    assert len(service.RAIN_REFERENCE) == 4
    assert all(e["rain_72h_mm"] > 250 for e in service.RAIN_REFERENCE)
