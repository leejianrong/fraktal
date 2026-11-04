from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.features import rasterize

from fraktal import biomes
from fraktal.coast import MANGROVE, TIDAL_FLAT, UNBUILDABLE
from fraktal.landcover_gen import WATER
from fraktal.procedural import generate_world, read_buildings, read_fields

CENTER = (104.0, 1.5)


@pytest.fixture(scope="module")
def coast(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, np.ndarray, rasterio.Affine]:
    path = tmp_path_factory.mktemp("coast")
    generate_world(path, CENTER, 2, biomes.resolve("mangrove_coast"))
    with rasterio.open(path / "landcover.tif") as src:
        return path, src.read(1), src.transform


def _near(mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            out[max(dr, 0) : h + min(dr, 0), max(dc, 0) : w + min(dc, 0)] |= mask[
                max(-dr, 0) : h + min(-dr, 0), max(-dc, 0) : w + min(-dc, 0)
            ]
    return out


def test_mangrove_sits_along_the_water_edge(coast: tuple[Path, np.ndarray, object]) -> None:
    _, land, _ = coast
    mangrove = land == MANGROVE
    assert 0.05 < mangrove.mean() < 0.35 and (land == TIDAL_FLAT).any()
    near_water = _near(land == WATER, 100)  # 100 cells at 5 m is 500 m
    assert (mangrove & near_water).sum() / mangrove.sum() > 0.95
    assert 0.2 < (land == WATER).mean() < 0.5  # a real sea on one side


def test_creeks_connect_to_the_sea(coast: tuple[Path, np.ndarray, object]) -> None:
    _, land, _ = coast
    water = land == WATER
    sea = np.zeros_like(water)
    sea[:, :] = False
    # the sea is the water body that spans the most of one tile edge; flood outward from it
    edges = [water[0, :], water[-1, :], water[:, 0], water[:, -1]]
    seed_edge = int(np.argmax([e.sum() for e in edges]))
    reached = np.zeros_like(water)
    if seed_edge == 0:
        reached[0, :] = water[0, :]
    elif seed_edge == 1:
        reached[-1, :] = water[-1, :]
    elif seed_edge == 2:
        reached[:, 0] = water[:, 0]
    else:
        reached[:, -1] = water[:, -1]
    while True:
        grown = (
            reached
            | np.pad(reached, 1)[:-2, 1:-1]
            | np.pad(reached, 1)[2:, 1:-1]
            | np.pad(reached, 1)[1:-1, :-2]
            | np.pad(reached, 1)[1:-1, 2:]
        ) & water
        if (grown == reached).all():
            break
        reached = grown
    creek_like = water & ~reached
    assert creek_like.sum() < 0.02 * water.sum()  # no water cut off from the sea


def test_no_buildings_on_water_or_wetland(coast: tuple[Path, np.ndarray, rasterio.Affine]) -> None:
    path, land, transform = coast
    buildings = read_buildings(path)
    footprint = rasterize(
        [(g, 1) for g in buildings.geometry], land.shape, transform=transform, dtype="uint8"
    )
    assert footprint.sum() > 0
    assert not np.isin(land[footprint > 0], UNBUILDABLE).any()


def test_fish_ponds_sit_behind_the_mangrove_and_off_water(
    coast: tuple[Path, np.ndarray, rasterio.Affine],
) -> None:
    path, land, transform = coast
    fields = read_fields(path)
    assert fields is not None and (fields["field_type"] == "fishpond").sum() > 5
    inside = rasterize(
        [(g, 1) for g in fields.geometry], land.shape, transform=transform, dtype="uint8"
    )
    assert not np.isin(land[inside > 0], (WATER, MANGROVE)).any()


def test_coast_world_is_byte_identical_per_seed(tmp_path: Path) -> None:
    config = biomes.resolve("mangrove_coast")
    one = generate_world(tmp_path / "a", CENTER, 4, config)
    two = generate_world(tmp_path / "b", CENTER, 4, config)
    assert one["world_id"] == two["world_id"]
    assert one["layers"] == two["layers"] and one["rasters"] == two["rasters"]
