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
