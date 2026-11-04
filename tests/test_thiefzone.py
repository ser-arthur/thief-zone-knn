import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thiefzone as tz


@pytest.fixture(scope="module")
def table():
    return tz.load_zone_table()


def test_table_has_ten_named_zones(table):
    assert len(table) == 10
    assert set(table["zone"]) == set(tz.HIERARCHY)


def test_upper_devonian_is_the_net_weighted_average_of_its_three_zones(table):
    zones = table.set_index("zone")
    children = zones.loc[["Devonian 3.3 (12)", "Devonian 3.2 (11)", "Devonian 3.1 (10)"]]
    rolled = tz.roll_up(children)
    reported = zones.loc["Upper Devonian"]
    assert rolled["h"] == pytest.approx(reported["h"], abs=0.01)
    for column in ["ng", "phif", "sw"]:
        assert rolled[column] == pytest.approx(reported[column], abs=0.002)
    assert rolled["k_ari"] == pytest.approx(reported["k_ari"], abs=1.0)
    assert rolled["k_geo"] == pytest.approx(reported["k_geo"], rel=0.1)
    assert rolled["k_har"] == pytest.approx(reported["k_har"], abs=0.06)


def test_devonian_row_is_reproduced_only_once_the_suspect_zone_fills_the_gap(table):
    column = tz.devonian_column(table)
    rolled = tz.roll_up(column)
    reported = table.set_index("zone").loc["Devonian"]
    assert rolled["h"] == pytest.approx(reported["h"], abs=0.01)
    for name in ["ng", "phif", "sw"]:
        assert rolled[name] == pytest.approx(reported[name], abs=0.002)
    assert rolled["k_ari"] == pytest.approx(reported["k_ari"], rel=0.01)
    assert rolled["k_geo"] == pytest.approx(reported["k_geo"], rel=0.1)
    assert rolled["k_har"] == pytest.approx(reported["k_har"], abs=0.06)

    without_suspect = tz.roll_up(column[column["zone"] != tz.SUSPECT_NAME])
    assert without_suspect["k_ari"] < 0.2 * reported["k_ari"]


def test_back_calculated_gap_matches_the_brief(table):
    gap = tz.reconstruct_gap(table)
    assert gap["top"] == pytest.approx(3454.7)
    assert gap["base"] == pytest.approx(3463.8)
    assert gap["k_ari"] == pytest.approx(tz.SUSPECT["k_ari"], rel=0.01)
    assert gap["ng"] == pytest.approx(tz.SUSPECT["ng"], abs=0.006)
    assert gap["phif"] == pytest.approx(tz.SUSPECT["phif"], abs=0.005)
    assert gap["sw"] == pytest.approx(tz.SUSPECT["sw"], abs=0.01)


def test_leaf_zones_tile_the_well_without_overlap(table):
    catalogue = tz.leaf_catalogue(table, include_zechstein=True)
    assert len(catalogue) == 8
    column = tz.devonian_column(table)
    pieces = [catalogue.iloc[:1], column, catalogue.iloc[5:]]
    tops = [t for piece in pieces for t in piece["top"]]
    bases = [b for piece in pieces for b in piece["base"]]
    assert len(tops) == 9
    for upper_base, lower_top in zip(bases[:-1], tops[1:]):
        assert math.isclose(upper_base, lower_top, abs_tol=1e-6)
    assert tops[0] == pytest.approx(3429.4)
    assert bases[-1] == pytest.approx(3682.0)


def test_flow_and_storage_shares_sum_to_one(table):
    profile = tz.flow_profile(tz.devonian_column(table))
    assert len(profile) == 5
    assert profile["flow_share"].sum() == pytest.approx(1.0)
    assert profile["storage_share"].sum() == pytest.approx(1.0)


def test_suspect_zone_dominates_devonian_flow_capacity(table):
    profile = tz.flow_profile(tz.devonian_column(table)).set_index("zone")
    suspect = profile.loc[tz.SUSPECT_NAME]
    assert suspect["flow_share"] == pytest.approx(0.904, abs=0.005)
    assert suspect["net_share"] < 0.2
    assert suspect["flow_to_storage"] > 4


@pytest.mark.parametrize("view", list(tz.FEATURE_VIEWS))
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("scaler", ["standard", "minmax"])
def test_knn_places_the_suspect_zone_in_the_devonian(table, view, k, scaler):
    catalogue = tz.leaf_catalogue(table, include_zechstein=(view == "stratigraphic"))
    prediction, neighbours = tz.locate(catalogue, tz.suspect_frame(), tz.FEATURE_VIEWS[view], k=k, scaler=scaler)
    assert prediction == "Devonian"
    assert len(neighbours) == len(catalogue)


def test_winland_port_classes():
    assert tz.port_class(float(tz.winland_r35(1280, 0.196))) == "megaport"
    assert tz.port_class(float(tz.winland_r35(108, 0.173))) == "macroport"
    assert tz.port_class(float(tz.winland_r35(0.3, 0.119))) == "microport"
