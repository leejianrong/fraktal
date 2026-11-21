"""Arid badland relief (FRK-187): terrain, no stray water, and banded rock on steep ground."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from fraktal import biomes, ground_soil
from fraktal.landcover_gen import WATER, slope_degrees
from fraktal.procedural import generate_world
from fraktal.terrain_gen import arid_terrain, fractal_terrain


def test_arid_terrain_is_deterministic_and_steeper_than_rolling() -> None:
    a = arid_terrain((200, 200), 5.0, seed=1, relief_m=90.0)
    assert np.array_equal(a, arid_terrain((200, 200), 5.0, seed=1, relief_m=90.0))
    assert not np.array_equal(a, arid_terrain((200, 200), 5.0, seed=2, relief_m=90.0))
    rolling = fractal_terrain((200, 200), 5.0, seed=1, relief_m=90.0)
    assert np.percentile(slope_degrees(a, 5.0), 90) > 2 * np.percentile(
        slope_degrees(rolling, 5.0), 90
    )
    assert a.dtype == np.float32


def test_a_badlands_world_has_no_open_water(tmp_path: Path) -> None:
    config = biomes.resolve("arid_badlands", {"size_m": 1000.0})
    generate_world(tmp_path / "w", (32.5, 15.6), 5, config)
    with rasterio.open(tmp_path / "w" / "landcover.tif") as d:
        assert not (d.read(1) == WATER).any()


def test_an_unknown_terrain_style_fails_loudly(tmp_path: Path) -> None:
    config = biomes.resolve("khartoum", {"terrain": {"style": "alpine"}})
    with pytest.raises(ValueError, match="terrain style"):
        generate_world(tmp_path / "w", (32.5, 15.6), 5, config)


def test_rock_shows_on_steep_bare_ground_only_in_a_rocky_biome() -> None:
    dem = np.tile(np.arange(40, dtype="float64") * 1.5, (40, 1))  # a steady 30 degree slope at 2 m
    ground = np.full((40, 40, 3), 0.7, dtype="float32")
    bare = np.ones((40, 40), dtype=bool)
    rocky = ground_soil.rock_relief(ground, dem, 2.0, "arid_badlands", bare)
    assert rocky.mean() < ground.mean() - 0.1
    assert ground_soil.rock_relief(ground, dem, 2.0, "khartoum", bare) is ground
    assert (ground_soil.rock_relief(ground, dem, 2.0, "arid_badlands", ~bare) == ground).all()
