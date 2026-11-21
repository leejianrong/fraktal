"""Highways and a cloverleaf (FRK-189, first slice)."""

from pathlib import Path

import pytest

from fraktal import biomes
from fraktal.procedural import generate_world, read_buildings, read_roads

CENTER = (-1.5, 52.0)


def _world(tmp_path: Path, seed: int = 5) -> Path:
    out = tmp_path / f"w{seed}"
    generate_world(out, CENTER, seed, biomes.resolve("river_plain"))
    return out


def test_a_highway_is_a_wide_arterial_with_four_ramps_at_its_crossing(tmp_path: Path) -> None:
    roads = read_roads(_world(tmp_path))
    highway, ramps = roads[roads.width_m == 28.0], roads[roads.width_m == 9.0]
    assert len(highway) >= 1 and set(highway.road_class) == {"arterial"}
    assert len(ramps) == 4
    centre = ramps.geometry.union_all().centroid
    assert highway.geometry.union_all().distance(centre) < 60.0  # the loops sit on the highway


def test_buildings_stay_off_the_highway_and_ramps(tmp_path: Path) -> None:
    world = _world(tmp_path)
    roads, houses = read_roads(world), read_buildings(world)
    wide = roads[roads.width_m >= 9.0]
    carriage = wide.geometry.buffer(wide.width_m / 2, cap_style="flat").union_all()
    assert houses.geometry.intersection(carriage).area.sum() < 1.0


def test_highways_off_by_default_and_deterministic(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    generate_world(plain, CENTER, 5, biomes.resolve("river_plain", {"highways": {"count": 0}}))
    assert not (read_roads(plain).width_m == 28.0).any()
    a, b = _world(tmp_path / "a"), _world(tmp_path / "b")
    assert (a / "roads.parquet").read_bytes() == (b / "roads.parquet").read_bytes()


def test_a_highway_bridges_water(tmp_path: Path) -> None:
    roads = read_roads(_world(tmp_path))
    bridges = roads[(roads.width_m == 28.0) & (roads.bridge > 0)]
    assert len(bridges) >= 1
    with pytest.raises(ValueError):
        biomes.resolve("river_plain", {"highways": {"count": "one"}})
