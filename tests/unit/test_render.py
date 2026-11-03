import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from fraktal import cli
from fraktal.render import AMBIENT, RenderConfig, region_grid, render_region, shade, utm_epsg
from fraktal.terrain import TerrainError
from fraktal.world import make_layer

CENTER = (32.525, 15.555)
SMALL = RenderConfig(
    gsd_m=1.0, tile_px=50, size_m=100.0, sun_azimuth_deg=270.0, sun_elevation_deg=45.0
)


@pytest.fixture
def terrain(tmp_path: Path) -> Path:
    """A flat 380 m DEM and all-grass land cover covering a small box around CENTER."""
    d = tmp_path / "terrain"
    d.mkdir()
    res = 1 / 12000
    left, top = 32.52, 15.56
    size = 120
    for name, data, dtype in [
        ("dem.tif", np.full((size, size), 380.0), "float32"),
        ("landcover.tif", np.full((size, size), 30), "uint8"),
    ]:
        with rasterio.open(
            d / name,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=from_origin(left, top, res, res),
        ) as dst:
            dst.write(data.astype(dtype), 1)
    return d


def one_building(center: tuple[float, float], size_m: float, height: float) -> gpd.GeoDataFrame:
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, utm_epsg(*center), always_xy=True).transform(*center)
    half = size_m / 2
    geom = box(x - half, y - half, x + half, y + half)
    layer = make_layer(
        "building",
        "w",
        [{"geometry": geom, "provenance": "procedural:test", "height_m": height}],
        f"EPSG:{utm_epsg(*center)}",
    )
    return layer


def test_utm_zone_for_khartoum() -> None:
    assert utm_epsg(32.5, 15.5) == 32636
    assert utm_epsg(-61.0, -10.0) == 32720


def test_grid_is_snapped_and_sized() -> None:
    epsg, transform, (tiles, side) = region_grid(CENTER, SMALL)
    assert (epsg, tiles, side) == (32636, 2, 100)
    assert transform.c % SMALL.gsd_m == 0 and transform.f % SMALL.gsd_m == 0


def test_region_renders_to_rgb_tiles_with_masks(tmp_path: Path, terrain: Path) -> None:
    out = tmp_path / "out"
    manifest = render_region(terrain, one_building(CENTER, 10, 10.0), CENTER, out, SMALL)
    assert manifest["tiles_per_side"] == 2
    tiles = sorted((out / "tiles").glob("r??_c??.tif"))
    assert len(tiles) == 4
    with rasterio.open(tiles[0]) as src:
        assert (src.count, src.width, src.height) == (3, 50, 50)
        assert src.crs.to_epsg() == 32636
    assert sum(t["building_pixels"] for t in manifest["tiles"]) == 100


def test_shadow_of_a_ten_metre_box_is_ten_metres_long(tmp_path: Path, terrain: Path) -> None:
    out = tmp_path / "out"
    render_region(terrain, one_building(CENTER, 10, 10.0), CENTER, out, SMALL)
    full = np.zeros((3, 100, 100), dtype="uint8")
    for r in range(2):
        for c in range(2):
            with rasterio.open(out / "tiles" / f"r{r:02d}_c{c:02d}.tif") as src:
                full[:, r * 50 : (r + 1) * 50, c * 50 : (c + 1) * 50] = src.read()
    grass = full[:, 0, 0]
    dark = (full[0] < grass[0] * 0.9) & (full[0] != 140)  # shadowed grass, not roof
    rows, cols = np.nonzero(dark)
    assert len(set(rows.tolist())) >= 10
    centre_row = int(np.median(rows))
    length = int(dark[centre_row].sum())
    assert abs(length - 10) <= 1  # 1 m pixels, sun in the west at 45 degrees: shadow falls east


def test_same_world_renders_to_byte_identical_output(tmp_path: Path, terrain: Path) -> None:
    buildings = one_building(CENTER, 12, 8.0)
    digests = []
    for name in ("a", "b"):
        out = tmp_path / name
        render_region(terrain, buildings, CENTER, out, SMALL)
        digests.append(
            {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "tiles").iterdir()}
            | {"manifest": hashlib.sha256((out / "render.json").read_bytes()).hexdigest()}
        )
    assert digests[0] == digests[1]


def test_region_outside_terrain_is_a_hard_error(tmp_path: Path, terrain: Path) -> None:
    far = (33.5, 15.555)
    with pytest.raises(TerrainError, match="does not cover"):
        render_region(terrain, one_building(far, 10, 5.0), far, tmp_path / "o", SMALL)


def test_shade_darkens_shadow_and_flags_unmapped_class() -> None:
    landcover = np.array([[30, 30], [0, 30]], dtype="uint8")
    mask = np.array([[0, 1], [0, 0]], dtype="uint8")
    shadow = np.array([[0, 0], [0, 1]], dtype="uint8")
    rgb = shade(landcover, mask, shadow)
    assert rgb.shape == (3, 2, 2)
    assert tuple(rgb[:, 0, 1]) == (140, 128, 115)  # roof
    assert tuple(rgb[:, 1, 0]) == (255, 0, 255)  # unmapped class is loud
    assert rgb[0, 1, 1] == round(140 * AMBIENT)


def test_cli_render_requires_terrain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["render", "--data-dir", str(tmp_path), "--buildings", str(tmp_path / "b.parquet")]
    assert cli.main([*args, "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["kind"] == "render_failed"
