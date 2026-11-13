"""Street trees along city roads (FRK-154)."""

from __future__ import annotations

from typing import Any

import numpy as np

from fraktal import trees3d

GSD = 1.0
SIDE = 400
ROAD = ([(20.0, 200.0), (380.0, 200.0)], 14.0)  # east-west, 14 m wide
CROSS = ([(200.0, 20.0), (200.0, 380.0)], 10.0)
FREE = np.zeros((SIDE, SIDE), dtype="uint8")


def _trees(
    roads: tuple[tuple[list[tuple[float, float]], float], ...] = (ROAD,),
    mask: np.ndarray[Any, Any] = FREE,
    biome: str = "manila_quezon",
    seed: int = 3,
    density: float = 1.0,
) -> list[list[Any]]:
    return trees3d.street_trees(list(roads), mask, GSD, biome, seed, density)


def test_biomes_without_a_street_style_get_none() -> None:
    assert _trees(biome="dense_urban") == []
    assert _trees(density=0.0) == []


def test_trees_line_both_sides_off_the_carriageway() -> None:
    trees = _trees()
    assert len(trees) > 20
    for _x, y, radius, height, species in trees:
        assert species == "canopy"  # Manila-Quezon
        assert abs(y - 200.0) >= 14.0 / 2  # never on the road itself
        assert 2.0 <= radius <= 9.0 and 6.0 <= height <= 16.0
    assert {y > 200.0 for _, y, *_ in trees} == {True, False}


def test_same_seed_same_trees_and_a_new_seed_differs() -> None:
    assert _trees(seed=3) == _trees(seed=3)
    assert _trees(seed=3) != _trees(seed=4)


def test_none_stand_on_a_building_or_at_a_junction() -> None:
    mask = FREE.copy()
    mask[:, :200] = 1  # everything west of x=200 is built up
    assert all(x >= 200.0 for x, *_ in _trees(mask=mask))
    junction = _trees(roads=(ROAD, CROSS))
    for x, y, *_ in junction:
        assert np.hypot(x - 200.0, y - 200.0) > 5.0 + 6.0  # clear of the crossing carriageways


def test_lanes_and_trees_outside_the_tile_are_skipped() -> None:
    lane = ([(20.0, 100.0), (380.0, 100.0)], 5.0)
    assert _trees(roads=(lane,)) == []
    edge = ([(20.0, 3.0), (380.0, 3.0)], 14.0)  # the south side falls outside the tile
    assert all(0.0 <= y < SIDE for _, y, *_ in _trees(roads=(edge,)))


def test_cities_differ_in_density() -> None:
    manila, khartoum = _trees(biome="manila_quezon"), _trees(biome="khartoum_city")
    assert len(manila) > 2 * len(khartoum) > 0


def test_a_dry_city_gets_dusty_crowns() -> None:
    assert {t[4] for t in _trees(biome="khartoum_city")} == {"canopy_dry"}
    assert {t[4] for t in _trees(biome="nairobi_centre")} == {"canopy"}
