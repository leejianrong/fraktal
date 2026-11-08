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
