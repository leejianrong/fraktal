"""Planned subdivisions (FRK-188): grid and cul-de-sac layouts with uniform lots."""

from pathlib import Path

import numpy as np
import pytest

from fraktal import biomes
from fraktal.procedural import generate_world, read_buildings, read_parcels, read_roads

CENTER = (-115.1, 36.1)


def _world(tmp_path: Path, biome: str, seed: int = 3, size_m: float = 700.0) -> Path:
    config = biomes.resolve(biome, {"size_m": size_m})
    out = tmp_path / f"{biome}-{seed}"
    generate_world(out, CENTER, seed, config)
    return out


@pytest.mark.parametrize("biome", ["planned_suburb", "planned_cul_de_sac"])
def test_lots_are_identical_rectangles_that_do_not_overlap(tmp_path: Path, biome: str) -> None:
    parcels = read_parcels(_world(tmp_path, biome))
    assert len(parcels) > 100
    areas = parcels.geometry.area.to_numpy()
    assert np.ptp(areas) < 1e-6 * areas.mean()
    assert np.allclose(parcels.geometry.area, parcels.geometry.minimum_rotated_rectangle().area)
    union = parcels.geometry.union_all()
    assert union.area == pytest.approx(areas.sum(), rel=1e-6)


@pytest.mark.parametrize("biome", ["planned_suburb", "planned_cul_de_sac"])
def test_houses_sit_inside_their_lots_and_off_the_streets(tmp_path: Path, biome: str) -> None:
    world = _world(tmp_path, biome)
    parcels, roads = read_parcels(world).set_index("id"), read_roads(world)
    houses = read_buildings(world)
    assert len(houses) > 0.8 * len(parcels)
    carriage = roads.geometry.buffer(roads.width_m / 2, cap_style="flat").union_all()
    for _, house in houses.iterrows():
        assert parcels.geometry[house.parcel_id].buffer(1e-6).contains(house.geometry)
        assert house.geometry.intersection(carriage).area < 0.5


def test_a_grid_has_only_two_street_directions_and_cul_de_sacs_have_bulbs(tmp_path: Path) -> None:
    grid = read_roads(_world(tmp_path, "planned_suburb"))
    angles = {
        round(np.degrees(np.arctan2(*np.subtract(g.coords[-1], g.coords[0])[::-1])) % 90, 1)
        for g in grid.geometry
    }
    assert len(angles) == 1
    cul = read_roads(_world(tmp_path, "planned_cul_de_sac"))
    assert (cul.width_m > 15).sum() > len(cul[cul.road_class == "arterial"])  # turning circles
    assert len(cul) > 0


def test_subdivisions_are_byte_identical_per_seed_and_differ_between_seeds(tmp_path: Path) -> None:
    a = (_world(tmp_path / "a", "planned_suburb") / "parcels.parquet").read_bytes()
    b = (_world(tmp_path / "b", "planned_suburb") / "parcels.parquet").read_bytes()
    c = (_world(tmp_path / "c", "planned_suburb", seed=4) / "parcels.parquet").read_bytes()
    assert a == b and a != c


def test_an_unknown_pattern_fails_loudly(tmp_path: Path) -> None:
    config = biomes.resolve("planned_suburb", {"subdivision": {"pattern": "spiral"}})
    with pytest.raises(ValueError, match="pattern"):
        generate_world(tmp_path / "w", CENTER, 3, config)
