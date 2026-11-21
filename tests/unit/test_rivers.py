"""Rivers and lakes (FRK-187)."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from fraktal import biomes
from fraktal.landcover_gen import BARE, GRASS, WATER
from fraktal.procedural import generate_world, read_roads
from fraktal.rivers import WaterConfig, paint_waters


def test_the_default_config_changes_nothing() -> None:
    dem = np.zeros((50, 50), dtype="float32")
    land = np.full((50, 50), BARE, dtype="uint8")
    out_dem, out = paint_waters(dem, land, 5.0, 1, WaterConfig(), 0.4)
    assert out_dem is dem and out is land


def test_rivers_and_lakes_are_water_carved_below_the_ground_with_green_banks() -> None:
    dem = np.full((200, 200), 300.0, dtype="float32")
    land = np.full((200, 200), BARE, dtype="uint8")
    cfg = WaterConfig(rivers=1, lakes=1)
    out_dem, out = paint_waters(dem, land, 5.0, 3, cfg, 0.4)
    water = out == WATER
    assert 0.02 < water.mean() < 0.4
    assert (out_dem[water] < 300.0).all() and (out_dem[~water] == 300.0).all()
    assert (out == GRASS).any()
    again = paint_waters(dem, land, 5.0, 3, cfg, 0.4)[1]
    assert np.array_equal(out, again)
    assert not np.array_equal(out, paint_waters(dem, land, 5.0, 4, cfg, 0.4)[1])


def test_dry_climates_get_no_green_bank() -> None:
    dem = np.full((200, 200), 300.0, dtype="float32")
    land = np.full((200, 200), BARE, dtype="uint8")
    out = paint_waters(dem, land, 5.0, 3, WaterConfig(rivers=1), 0.9)[1]
    assert not (out == GRASS).any()


def test_a_river_plain_world_has_water_and_roads_bridge_it(tmp_path: Path) -> None:
    generate_world(tmp_path / "w", (-1.5, 52.0), 5, biomes.resolve("river_plain"))
    with rasterio.open(tmp_path / "w" / "landcover.tif") as d:
        assert (d.read(1) == WATER).mean() > 0.02
    assert len(read_roads(tmp_path / "w")) > 0


def test_a_river_world_is_byte_identical_per_seed(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        generate_world(out, (-1.5, 52.0), 5, biomes.resolve("river_plain", {"size_m": 1000.0}))
    assert (a / "landcover.tif").read_bytes() == (b / "landcover.tif").read_bytes()
    assert (a / "dem.tif").read_bytes() == (b / "dem.tif").read_bytes()
    with pytest.raises(ValueError):
        biomes.resolve("river_plain", {"water": {"rivers": "two"}})
