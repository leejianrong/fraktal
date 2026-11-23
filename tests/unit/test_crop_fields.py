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


def test_blocks_are_cut_into_narrow_strips_with_tracks_on_some_edges(tmp_path: Path) -> None:
    from fraktal.fields import TRACK_WIDTH_M
    from fraktal.render import ROAD_TRACK

    fields = read_fields(_world(tmp_path))
    assert fields is not None
    crops, tracks = fields[fields.field_type == "crop"], fields[fields.field_type == "track"]
    assert len(tracks) > 3
    widths = crops.geometry.minimum_rotated_rectangle().apply(
        lambda r: min(
            np.hypot(*np.subtract(r.exterior.coords[i + 1], r.exterior.coords[i])) for i in range(2)
        )
    )
    assert widths.median() < 35.0  # strips, not the 80 m or wider blocks
    assert np.allclose(tracks.geometry.area / tracks.geometry.length, TRACK_WIDTH_M / 2, rtol=0.1)
    from affine import Affine

    t = tracks.iloc[0]
    w, n = t.geometry.bounds[0] - 20, t.geometry.bounds[3] + 20
    side = int(max(t.geometry.bounds[2] - w, n - t.geometry.bounds[1]) + 40)
    painted = _field_raster(
        fields.iloc[[t.name]], Affine(1.0, 0, w, 0, -1.0, n), side, np.zeros((side, side))
    )
    assert (painted == ROAD_TRACK).any()


def test_industrial_halls_include_blue_and_white_sheet_roofs() -> None:
    from fraktal.building_grammar import HALL_ROOF_WEIGHTS, HALL_ROOFS

    assert abs(sum(HALL_ROOF_WEIGHTS) - 1.0) < 1e-9
    share = dict(zip(HALL_ROOFS, HALL_ROOF_WEIGHTS, strict=True))
    assert share["roof-tin-blue"] + share["roof-tin-white"] >= 0.5
    assert biomes.resolve("farmed_plain").industrial.sites_per_km > 0
