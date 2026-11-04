import json
from pathlib import Path

import numpy as np

from fraktal import biomes
from fraktal.blender_export import export_scene
from fraktal.procedural import generate_world

CENTER = (104.0, 1.5)


def test_export_is_deterministic_and_in_tile_coordinates(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban"))
    a = export_scene(world, tmp_path / "a", CENTER, 200.0, 1.0)
    b = export_scene(world, tmp_path / "b", CENTER, 200.0, 1.0)
    assert (tmp_path / "a" / "scene.json").read_text() == (
        tmp_path / "b" / "scene.json"
    ).read_text()
    assert a["px"] == 200 and a["buildings"]
    ground = np.load(tmp_path / "a" / "ground.npy")
    assert ground.shape == (200, 200, 3) and ground.min() >= 0.0 and ground.max() <= 1.0
    for building in a["buildings"]:  # rings are metres from the south-west corner
        xs = [p[0] for p in building["ring"]]
        assert building["height_m"] > 0 and max(xs) > -50 and min(xs) < 250
    assert json.loads((tmp_path / "a" / "scene.json").read_text())["schema"] == 1
    assert b["buildings"] == a["buildings"]


def test_export_has_textured_ground_and_trees_in_tile(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban"))
    scene = export_scene(world, tmp_path / "t", CENTER, 200.0, 1.0)
    assert scene["trees"], "dense_urban tiles still have some trees"
    for x, y, radius, height in scene["trees"]:
        assert 0 <= x <= 200 and 0 <= y <= 200 and 0 < radius < 6 and 3 < height < 10
    flat = export_scene(world, tmp_path / "f", CENTER, 200.0, 1.0, texture=0.0, tree_density=0.0)
    assert flat["trees"] == []
    textured = np.load(tmp_path / "t" / "ground.npy")
    plain = np.load(tmp_path / "f" / "ground.npy")
    assert textured.std() > plain.std()  # the texture adds variation within a class
