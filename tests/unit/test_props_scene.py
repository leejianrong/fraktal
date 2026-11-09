import math

import numpy as np
import pytest

from fraktal import labels3d, props_scene
from fraktal.render import ROAD_LOCAL

SIDE, GSD = 400, 0.5  # a 200 m tile


def _world() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    surface = np.full((SIDE, SIDE), 30, dtype="uint8")  # grass
    mask = np.zeros((SIDE, SIDE), dtype="uint8")
    mask[100:200, 100:300] = 1  # a big building
    road = np.zeros((SIDE, SIDE), dtype="uint8")
    road[300:312, :] = ROAD_LOCAL  # a street across the tile
    return surface, mask, road


def test_sampling_is_deterministic_and_seeded() -> None:
    surface, mask, road = _world()
    a = props_scene.sample(surface, mask, road, GSD, SIDE, 400.0, 7)
    assert a == props_scene.sample(surface, mask, road, GSD, SIDE, 400.0, 7)
    assert a != props_scene.sample(surface, mask, road, GSD, SIDE, 400.0, 8)
    assert 5 <= len(a) <= 16  # 400 per km2 on 0.04 km2


def test_props_keep_off_buildings_and_tuk_tuks_stay_on_the_street() -> None:
    surface, mask, road = _world()
    props = props_scene.sample(surface, mask, road, GSD, SIDE, 2000.0, 3)
    assert props
    for p in props:
        row, col = SIDE - 1 - int(p["y"] / GSD), int(p["x"] / GSD)
        assert mask[row, col] == 0
        on_street = road[row, col] == ROAD_LOCAL
        assert on_street == (p["kind"] == "tuk_tuk")


def test_classes_come_from_the_catalogue() -> None:
    surface, mask, road = _world()
    for p in props_scene.sample(surface, mask, road, GSD, SIDE, 2000.0, 5):
        assert p["class"] == labels3d.CLASSES[props_scene.PROP_CLASS[p["kind"]]]
    assert labels3d.CLASSES["yard_object"] == 9 and labels3d.CLASSES["wall"] == 8


def test_no_roads_means_no_tuk_tuks_and_zero_density_means_none() -> None:
    surface, mask, _ = _world()
    kinds = {p["kind"] for p in props_scene.sample(surface, mask, None, GSD, SIDE, 2000.0, 1)}
    assert "tuk_tuk" not in kinds
    assert props_scene.sample(surface, mask, None, GSD, SIDE, 0.0, 1) == []


def test_pack_gives_consistent_arrays_with_every_face_owned() -> None:
    surface, mask, road = _world()
    props = props_scene.sample(surface, mask, road, GSD, SIDE, 800.0, 2)
    packed = props_scene.pack(props)
    n_faces = len(packed["faces"])
    assert packed["faces"].max() < len(packed["vertices"])
    assert len(packed["face_material"]) == len(packed["face_owner"]) == n_faces
    assert set(packed["face_owner"].tolist()) == set(range(len(props)))
    assert packed["face_material"].max() < len(packed["materials"])
    assert np.array_equal(packed["vertices"], props_scene.pack(props)["vertices"])  # deterministic
    assert props_scene.pack([])["faces"].shape[0] == 0


def test_prop_instance_base_matches_the_blender_script() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[2] / "src/fraktal/blender_scene.py").read_text()
    import re

    found = re.search(r"PROP_INSTANCE_BASE = \(?\s*(\d+)", source)
    assert found and int(found.group(1)) == labels3d.PROP_INSTANCE_BASE
    assert labels3d.meta([], 0, props=3)["props_first_instance"] == labels3d.PROP_INSTANCE_BASE + 1


def test_variation_differs_by_seed_and_stays_in_range() -> None:
    surface, mask, road = _world()
    a = props_scene.sample(surface, mask, road, GSD, SIDE, 2000.0, 11)
    b = props_scene.sample(surface, mask, road, GSD, SIDE, 2000.0, 12)
    assert [p["scale"] for p in a] != [p["scale"] for p in b]
    assert len({(p["tint"], p["wear"]) for p in a}) > 3
    lo, hi = props_scene.SCALE_LIMITS
    for p in a:
        assert lo <= p["scale"] <= hi
        assert p["tint"] in range(len(props_scene.TINT)) and p["wear"] in range(3)


def test_varying_the_look_does_not_move_a_prop() -> None:
    """Scale, tint and wear come from their own stream, so where props go is unchanged."""
    surface, mask, road = _world()
    a = props_scene.sample(surface, mask, road, GSD, SIDE, 400.0, 7)
    assert [(p["kind"], p["variant"], p["x"], p["y"], p["yaw"]) for p in a[:3]] == [
        ("signboard", 0, 60.8, 51.23, 0.6483),
        ("signboard", 3, 98.95, 22.23, 4.9456),
        ("market_stall", 3, 46.28, 33.93, 2.5043),
    ]


def test_golden_sampler_output() -> None:
    surface, mask, road = _world()
    a = props_scene.sample(surface, mask, road, GSD, SIDE, 400.0, 7)
    assert len(a) == 16
    assert a[0] == {
        "kind": "signboard", "variant": 0, "x": 60.8, "y": 51.23, "yaw": 0.6483,
        "scale": 0.975, "tint": 3, "wear": 0, "class": 9,
    }  # fmt: skip
    assert props_scene.repeat_stats(a) == {
        "count": 16, "distinct_shapes": 15, "distinct_looks": 16,
        "shape_repeat_rate": 0.062, "look_repeat_rate": 0.0,
    }  # fmt: skip


def test_pack_applies_scale_and_looks() -> None:
    base = {"kind": "container", "variant": 0, "x": 0.0, "y": 0.0, "yaw": 0.0, "class": 9}
    plain = props_scene.pack([{**base, "scale": 1.0, "tint": 2, "wear": 0}])
    big = props_scene.pack([{**base, "scale": 1.08, "tint": 2, "wear": 0}])
    assert abs(big["vertices"][:, 0].max() / plain["vertices"][:, 0].max() - 1.08) < 1e-5
    worn = props_scene.pack([{**base, "scale": 1.0, "tint": 2, "wear": 2}])
    first_plain, first_worn = plain["materials"][0], worn["materials"][0]
    assert first_plain[0] != first_worn[0]  # a different look is a different material
    assert first_worn[2] > first_plain[2]  # rougher
    assert sum(first_worn[1]) < sum(first_plain[1])  # darker
    two = props_scene.pack(
        [{**base, "scale": 1.0, "tint": 0, "wear": 0}, {**base, "scale": 1.0, "tint": 3, "wear": 1}]
    )
    assert len({m[0] for m in two["materials"]}) == len(two["materials"])  # no duplicates


def test_props_keep_off_compound_walls() -> None:
    surface = np.full((400, 400), 60, dtype="uint8")
    mask = np.zeros((400, 400), dtype="uint8")
    wall = [[20.0, 100.0, 180.0, 100.0]]
    plain = props_scene.sample(surface, mask, None, 0.5, 400, 4000.0, 3)
    walled = props_scene.sample(surface, mask, None, 0.5, 400, 4000.0, 3, walls=wall)
    assert any(abs(p["y"] - 100.0) < 1.0 and 20 < p["x"] < 180 for p in plain)
    assert not any(abs(p["y"] - 100.0) < 0.6 and 20 < p["x"] < 180 for p in walled)


def test_street_pass_adds_bikes_vans_people_and_litter_without_moving_the_first_pass() -> None:
    surface, mask, road = _world()
    first = props_scene.sample(surface, mask, road, GSD, SIDE, 800.0, 2)
    street = props_scene.sample(
        surface, mask, road, GSD, SIDE, 1200.0, 2,
        mix=props_scene.STREET_MIX, stream="props:street", avoid=first,
    )  # fmt: skip
    assert props_scene.sample(surface, mask, road, GSD, SIDE, 800.0, 2) == first
    assert {p["kind"] for p in street} <= set(props_scene.STREET_MIX)
    assert {"motorbike", "pedestrian"} <= {p["kind"] for p in street}
    for p in street:
        assert p["class"] == labels3d.CLASSES[props_scene.PROP_CLASS[p["kind"]]]
    classes = {labels3d.CLASSES["vehicle"], labels3d.CLASSES["person"]}
    assert classes <= {p["class"] for p in street}
    for p in street:  # none lands on top of a first-pass prop
        assert all(math.hypot(p["x"] - q["x"], p["y"] - q["y"]) >= 4.0 for q in first)


def _prop(kind: str, x: float, y: float) -> dict[str, object]:
    return {"kind": kind, "variant": 0, "x": x, "y": y, "yaw": 1.0, "scale": 1.0, "tint": 0,
            "wear": 0, "class": labels3d.CLASSES[props_scene.PROP_CLASS[kind]]}  # fmt: skip


def test_vehicles_face_along_the_road_and_people_stand_on_the_footpath() -> None:
    from shapely.geometry import LineString

    mask = np.zeros((SIDE, SIDE), dtype="uint8")
    roads = [(LineString([(0.0, 100.0), (200.0, 100.0)]), 8.0)]  # east-west, 8 m wide
    props = [
        _prop("minibus", 50.0, 103.0), _prop("motorbike", 80.0, 98.0),
        _prop("pedestrian", 120.0, 101.0), _prop("container", 60.0, 60.0),
    ]  # fmt: skip
    out = props_scene.align_to_roads(props, roads, mask, GSD, SIDE, 3)
    van, bike, walker, box = out
    for v in (van, bike):
        assert abs(math.sin(v["yaw"])) < 0.15  # along the road, either way
        assert abs(v["y"] - 100.0) < 4.0  # still on the carriageway, towards the kerb
    assert van["y"] > 100.0 > bike["y"]  # each stays on its own side
    assert abs(walker["y"] - 100.0) == pytest.approx(4.9, abs=0.01)  # beyond the edge
    assert box == props[3]  # not a street kind: untouched
    assert props_scene.align_to_roads(props, roads, mask, GSD, SIDE, 3) == out
    assert props_scene.align_to_roads(props, [], mask, GSD, SIDE, 3) == props


def test_a_person_does_not_move_into_a_building() -> None:
    from shapely.geometry import LineString

    mask = np.ones((SIDE, SIDE), dtype="uint8")  # buildings everywhere
    roads = [(LineString([(0.0, 100.0), (200.0, 100.0)]), 8.0)]
    given = [_prop("pedestrian", 120.0, 101.0)]
    (walker,) = props_scene.align_to_roads(given, roads, mask, GSD, SIDE, 3)
    assert (walker["x"], walker["y"]) == (120.0, 101.0)
