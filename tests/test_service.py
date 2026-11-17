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
