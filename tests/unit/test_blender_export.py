import json
from pathlib import Path

import numpy as np
import pytest

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
    for x, y, radius, height, species in scene["trees"]:
        assert 0 <= x <= 200 and 0 <= y <= 200 and 0 < radius < 6 and 3 < height < 10
        assert species in {"broadleaf", "palm", "shrub"} and (height >= 3.5 or species == "shrub")
    flat = export_scene(world, tmp_path / "f", CENTER, 200.0, 1.0, texture=0.0, tree_density=0.0)
    assert flat["trees"] == []
    textured = np.load(tmp_path / "t" / "ground.npy")
    plain = np.load(tmp_path / "f" / "ground.npy")
    assert textured.std() > plain.std()  # the texture adds variation within a class


def test_every_mesh_face_knows_its_building(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("dense_urban"))
    scene = export_scene(world, tmp_path / "s", CENTER, 200.0, 1.0)
    owner = np.load(tmp_path / "s" / "mesh_face_building.npy")
    assert len(owner) == len(np.load(tmp_path / "s" / "mesh_face_sizes.npy"))
    assert owner.min() >= 0 and owner.max() < len(scene["buildings"])
    assert np.all(np.diff(owner) >= 0)  # faces are grouped by building


def test_the_3d_heights_follow_the_biome_and_world_keeps_the_original(tmp_path: Path) -> None:
    world = tmp_path / "w"
    generate_world(world, CENTER, 3, biomes.resolve("khartoum"))
    auto = export_scene(world, tmp_path / "a", CENTER, 400.0, 1.0)
    kept = export_scene(world, tmp_path / "k", CENTER, 400.0, 1.0, height_profile="world")
    tall = [b["height_m"] for b in auto["buildings"]]
    assert tall and max(tall) < 5 * 3.0 + 0.8 + 1e-9
    assert max(b["height_m"] for b in kept["buildings"]) > max(tall)
    again = export_scene(world, tmp_path / "b", CENTER, 400.0, 1.0)
    assert again["buildings"] == auto["buildings"]


def _world(tmp_path: Path, biome: str) -> Path:
    world = tmp_path / biome
    generate_world(world, CENTER, 3, biomes.resolve(biome))
    return world


def test_water_is_glossy_in_the_roughness_map(tmp_path: Path) -> None:
    world = _world(tmp_path, "mangrove_coast")
    export_scene(world, tmp_path / "m", CENTER, 800.0, 1.0)
    rough = np.load(tmp_path / "m" / "ground_rough.npy")
    ground = np.load(tmp_path / "m" / "ground.npy")
    assert rough.shape == ground.shape[:2] and rough.min() < 0.1 and rough.max() > 0.9
    assert 0.05 < float((rough < 0.1).mean()) < 0.9  # some, not all, of the tile is open water


def test_oil_palm_rows_become_palms_in_regular_rows(tmp_path: Path) -> None:
    world = _world(tmp_path, "oil_palm_plantation")
    scene = export_scene(world, tmp_path / "o", CENTER, 800.0, 1.0)
    palms = [t for t in scene["trees"] if t[4] == "palm"]
    assert all(0 <= t[0] <= 800 and 0 <= t[1] <= 800 for t in palms)
    assert sum(t[3] > 7.9 for t in palms) > 200  # planted rows are 8 to 13 m, scattered trees less
    again = export_scene(world, tmp_path / "o2", CENTER, 800.0, 1.0)
    assert again["trees"] == scene["trees"]


def test_paddy_fields_change_the_ground_colour(tmp_path: Path) -> None:
    world = _world(tmp_path, "paddy_terraces")
    export_scene(world, tmp_path / "p", CENTER, 800.0, 1.0)
    colours = np.load(tmp_path / "p" / "ground.npy").reshape(-1, 3)
    greens = colours[(colours[:, 1] > 0.55) & (colours[:, 0] < 0.5)]  # bright young rice
    assert len(greens) > 1000


def test_cars_stand_on_roads_headed_along_them_and_poles_line_wide_streets(tmp_path: Path) -> None:
    world = tmp_path / "khartoum"
    generate_world(world, CENTER, 3, biomes.resolve("khartoum"))
    scene = export_scene(world, tmp_path / "v", CENTER, 400.0, 1.0)
    assert len(scene["vehicles"]) > 5 and len(scene["poles"]) > 5
    for x, y, heading, colour in scene["vehicles"]:
        assert 0 <= x <= 400 and 0 <= y <= 400 and -7 < heading < 7 and colour in range(5)
    for x, y, height in scene["poles"]:
        assert 0 <= x <= 400 and 0 <= y <= 400 and height == 8.0
    again = export_scene(world, tmp_path / "v2", CENTER, 400.0, 1.0)
    assert (again["vehicles"], again["poles"]) == (scene["vehicles"], scene["poles"])
    none = export_scene(world, tmp_path / "v3", CENTER, 400.0, 1.0, vehicle_density=0.0)
    assert none["vehicles"] == [] and none["poles"] == scene["poles"]


def test_a_car_heading_follows_the_road_direction(tmp_path: Path) -> None:
    from shapely.geometry import Point

    from fraktal.procedural import read_roads

    world = tmp_path / "khartoum"
    generate_world(world, CENTER, 3, biomes.resolve("khartoum"))
    scene = export_scene(world, tmp_path / "h", CENTER, 400.0, 1.0)
    roads = read_roads(world).to_crs(scene["crs"])
    west, south = scene["origin_sw"]
    lines = [(r.geometry, r.width_m) for r in roads.itertuples()]
    off = []
    for x, y, heading, _ in scene["vehicles"]:
        p = Point(x + west, y + south)
        line, _ = min(lines, key=lambda lw: lw[0].distance(p))
        d = line.project(p)
        a, b = line.interpolate(max(d - 2, 0)), line.interpolate(d + 2)
        along = np.arctan2(b.y - a.y, b.x - a.x)
        diff = abs((heading - along + np.pi / 2) % np.pi - np.pi / 2)  # a car may face either way
        off.append(diff)
    assert float(np.median(off)) < 0.2  # radians, about 11 degrees


def test_the_ground_helpers_bend_and_soften_class_edges() -> None:
    import numpy as np

    from fraktal import blender_export as be

    blocks = np.kron(np.array([[10, 30], [30, 40]], dtype="uint8"), np.ones((40, 40), "uint8"))
    bent = be._bend_edges(blocks, 5, 40.0)
    assert bent.shape == blocks.shape and set(np.unique(bent)) <= {10, 30, 40}
    assert np.array_equal(bent, be._bend_edges(blocks, 5, 40.0))
    assert not np.array_equal(bent, be._bend_edges(blocks, 6, 40.0))
    assert (bent != blocks).mean() > 0.02  # the edges moved

    step = np.zeros((30, 30, 3), "float32")
    step[:, 15:] = 1.0
    soft = be._blur(step, 3.0)
    assert 0.0 < soft[10, 15, 0] < 1.0 and soft[10, 2, 0] < 0.05 and soft[10, 27, 0] > 0.95
    assert np.allclose(be._blur(np.full((20, 20, 3), 0.4, "float32"), 3.0), 0.4, atol=1e-5)


def test_terrain_nodes_follow_the_dem_with_north_up_turned_to_y_up() -> None:
    from fraktal.blender_export import terrain_nodes

    dem = np.zeros((100, 100))
    dem[:50] = 10.0  # the northern half is 10 m higher (row 0 is north)
    nodes, cell, low = terrain_nodes(dem + 300.0, 1.0, 100.0)
    assert nodes.shape == (101, 101) and cell == 1.0 and low == pytest.approx(300.0)
    assert nodes[100, 50] == pytest.approx(10.0) and nodes[0, 50] == pytest.approx(0.0)
    assert nodes[:, 0].tolist() == pytest.approx(nodes[:, 100].tolist())
    ramp = np.tile(np.arange(100.0), (100, 1))  # rises to the east
    east, *_ = terrain_nodes(ramp, 1.0, 100.0)
    assert east[50, 100] > east[50, 0] + 90.0
    big, big_cell, _ = terrain_nodes(np.zeros((4000, 4000)), 0.5, 2000.0)  # capped, so coarser
    assert big.shape == (701, 701) and big_cell > 2.0


def test_a_hilly_tile_exports_a_height_grid_and_a_flat_tile_does_not(tmp_path: Path) -> None:
    world = _world(tmp_path, "paddy_terraces")
    scene = export_scene(world, tmp_path / "t", CENTER, 800.0, 1.0)
    nodes = np.load(tmp_path / "t" / scene["terrain"]["file"])
    assert nodes.min() == 0.0 and nodes.max() == pytest.approx(scene["terrain"]["relief_m"], abs=0.01)
    assert nodes.max() > 3.0  # terraces and hills stand clearly above the lowest ground
    assert nodes.shape[0] == round(800.0 / scene["terrain"]["cell_m"]) + 1
    again = export_scene(world, tmp_path / "t2", CENTER, 800.0, 1.0)
    assert np.array_equal(np.load(tmp_path / "t2" / "ground_height.npy"), nodes)
    assert again["terrain"] == scene["terrain"]
