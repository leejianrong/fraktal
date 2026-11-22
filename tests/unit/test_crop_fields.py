"""Rectilinear crop fields and canals (FRK-136, FRK-187)."""

from pathlib import Path

import numpy as np
import pytest

from fraktal import biomes
from fraktal.fields import CANAL_WIDTH_M
from fraktal.procedural import generate_world, read_buildings, read_fields, read_roads
from fraktal.render import CANAL_WATER, CROP_CODES, CROP_STRIPE, _crop_rows, _field_raster

CENTER = (-1.5, 52.0)


def _world(tmp_path: Path, seed: int = 5) -> Path:
    out = tmp_path / f"w{seed}"
    generate_world(out, CENTER, seed, biomes.resolve("farmed_plain"))
    return out


def test_crop_blocks_are_rectangles_off_buildings_and_roads_with_canals_on_edges(
    tmp_path: Path,
) -> None:
    world = _world(tmp_path)
    fields = read_fields(world)
    assert fields is not None
    crops, canals = fields[fields.field_type == "crop"], fields[fields.field_type == "canal"]
    assert len(crops) > 10 and len(canals) > 2
    assert set(crops.state) <= {"green", "gold", "fallow", "plough", "stubble"}
    assert np.allclose(
        crops.geometry.area, crops.geometry.minimum_rotated_rectangle().area, rtol=1e-3
    )
    roads, houses = read_roads(world), read_buildings(world)
    carriage = roads.geometry.buffer(roads.width_m / 2, cap_style="flat").union_all()
    assert crops.geometry.intersection(carriage).area.sum() < 1.0
    assert crops.geometry.intersection(houses.geometry.union_all()).area.sum() < 1.0
    blocks = crops.geometry.union_all()
    assert all(
        c.intersection(blocks.buffer(CANAL_WIDTH_M)).area > 0.5 * c.area for c in canals.geometry
    )


def test_rows_stripe_a_crop_field_in_its_own_colour_and_canals_paint_water(tmp_path: Path) -> None:
    from affine import Affine

    fields = read_fields(_world(tmp_path))
    assert fields is not None
    crop = fields[fields.field_type == "crop"].iloc[0]
    west, north = crop.geometry.bounds[0] - 50, crop.geometry.bounds[3] + 50
    side = int(max(crop.geometry.bounds[2] - west, north - crop.geometry.bounds[1]) + 100)
    transform = Affine(1.0, 0, west, 0, -1.0, north)
    painted = _field_raster(fields.iloc[[crop.name]], transform, side, np.zeros((side, side)))
    base = CROP_CODES[str(crop.state)]
    assert (painted == base).any() and (painted == base + CROP_STRIPE).any()
    canal = fields[fields.field_type == "canal"].iloc[0]
    cw, cn = canal.geometry.bounds[0] - 20, canal.geometry.bounds[3] + 20
    cside = int(max(canal.geometry.bounds[2] - cw, cn - canal.geometry.bounds[1]) + 40)
    water = _field_raster(
        fields.iloc[[canal.name]], Affine(1.0, 0, cw, 0, -1.0, cn), cside, np.zeros((cside, cside))
    )
    assert (water == CANAL_WATER).sum() > 10
    assert _crop_rows(np.zeros((4, 4), dtype="uint8"), crop, transform).shape == (4, 4)


def test_crop_fields_are_byte_identical_per_seed_and_off_by_default(tmp_path: Path) -> None:
    a, b = _world(tmp_path / "a"), _world(tmp_path / "b")
    assert (a / "fields.parquet").read_bytes() == (b / "fields.parquet").read_bytes()
    plain = biomes.resolve("khartoum")
    assert plain.fields.crop_zones == 0
    with pytest.raises(ValueError):
        biomes.resolve("farmed_plain", {"fields": {"canal_share": "many"}})
