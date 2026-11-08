import numpy as np

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
