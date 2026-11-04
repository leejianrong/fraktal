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
from fraktal.render import (
    AMBIENT,
    RenderConfig,
    region_grid,
    render_region,
    roof_table,
    shade,
    utm_epsg,
)
from fraktal.terrain import TerrainError
from fraktal.world import make_layer

CENTER = (32.525, 15.555)
SMALL = RenderConfig(
    gsd_m=1.0, tile_px=50, size_m=100.0, sun_azimuth_deg=270.0, sun_elevation_deg=45.0
)


@pytest.fixture
def terrain(tmp_path: Path) -> Path:
    return terrain_at(tmp_path / "terrain", CENTER)


def terrain_at(d: Path, center: tuple[float, float], land_class: int = 30) -> Path:
    """A flat 380 m DEM and uniform land cover (about 1.1 km square) around ``center``."""
    d.mkdir(parents=True)
    res = 1 / 12000
    size = 120
    left, top = center[0] - size * res / 2, center[1] + size * res / 2
    for name, data, dtype in [
        ("dem.tif", np.full((size, size), 380.0), "float32"),
        ("landcover.tif", np.full((size, size), land_class), "uint8"),
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


def test_label_masks_equal_the_independent_rasterisation(tmp_path: Path, terrain: Path) -> None:
    from fraktal.labels import independent_building_mask

    buildings = one_building(CENTER, 13.3, 9.0)
    out = tmp_path / "out"
    manifest = render_region(terrain, buildings, CENTER, out, SMALL)
    assert manifest["dropped"] == []
    epsg, transform, _ = region_grid(CENTER, SMALL)
    local = buildings.to_crs(epsg)
    for rec in manifest["tiles"]:
        with rasterio.open(out / "tiles" / f"{rec['tile']}_buildings.tif") as src:
            expected = independent_building_mask(list(local.geometry), src.transform, src.shape)
            assert (src.read(1) == expected).all()
    assert sum(r["building_pixels"] for r in manifest["tiles"]) > 0


def test_mask_and_image_share_crs_transform_and_size(tmp_path: Path, terrain: Path) -> None:
    out = tmp_path / "out"
    render_region(terrain, one_building(CENTER, 10, 5.0), CENTER, out, SMALL)
    for stem in ("r00_c00", "r01_c01"):
        grids = []
        for suffix in ("", "_buildings", "_landcover"):
            with rasterio.open(out / "tiles" / f"{stem}{suffix}.tif") as src:
                grids.append((src.crs, src.transform, src.shape))
        assert len(set(grids)) == 1


def test_a_corrupted_tile_is_dropped_and_listed(
    tmp_path: Path, terrain: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fraktal import render as render_module

    real = render_module._building_rasters

    def corrupt(buildings: gpd.GeoDataFrame, transform: object, side: int) -> tuple[object, ...]:
        height, mask, index = real(buildings, transform, side)
        mask[3, 3] ^= 1  # one wrong pixel in tile r00_c00
        return height, mask, index

    monkeypatch.setattr(render_module, "_building_rasters", corrupt)
    out = tmp_path / "out"
    manifest = render_region(terrain, one_building(CENTER, 10, 5.0), CENTER, out, SMALL)
    assert [d["tile"] for d in manifest["dropped"]] == ["r00_c00"]
    assert "independent rasterisation" in manifest["dropped"][0]["reason"]
    assert "r00_c00" not in {t["tile"] for t in manifest["tiles"]}
    assert not list((out / "tiles").glob("r00_c00*"))
    assert len(manifest["tiles"]) == 3


def test_independent_mask_rejects_rotated_grids() -> None:
    from rasterio.transform import Affine

    from fraktal.labels import independent_building_mask

    with pytest.raises(ValueError, match="north-up"):
        independent_building_mask([box(0, 0, 1, 1)], Affine(1, 0.1, 0, 0, -1, 0), (2, 2))


def test_variation_zero_keeps_the_flat_v1_look(tmp_path: Path, terrain: Path) -> None:
    out = tmp_path / "flat"
    render_region(terrain, one_building(CENTER, 10, 5.0), CENTER, out, SMALL)
    with rasterio.open(out / "tiles" / "r00_c00.tif") as src:
        rgb = src.read()
    assert len({tuple(rgb[:, y, x]) for y in range(50) for x in range(50)}) <= 3


def varied(seed: int, variation: float = 0.2) -> RenderConfig:
    return RenderConfig(
        gsd_m=1.0, tile_px=50, size_m=100.0, sun_azimuth_deg=270.0,
        sun_elevation_deg=45.0, seed=seed, variation=variation,
    )  # fmt: skip


def read_hash(out: Path) -> str:
    return hashlib.sha256((out / "tiles" / "r00_c00.tif").read_bytes()).hexdigest()


def test_varied_render_is_deterministic_and_seed_dependent(tmp_path: Path, terrain: Path) -> None:
    layer = one_building(CENTER, 10, 5.0)
    hashes = []
    for name, seed in [("a", 1), ("b", 1), ("c", 2)]:
        out = tmp_path / name
        render_region(terrain, layer, CENTER, out, varied(seed))
        hashes.append(read_hash(out))
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_variation_leaves_the_labels_untouched(tmp_path: Path, terrain: Path) -> None:
    layer = one_building(CENTER, 10, 5.0)
    flat = render_region(terrain, layer, CENTER, tmp_path / "f", SMALL)
    var = render_region(terrain, layer, CENTER, tmp_path / "v", varied(3))
    assert [t["mask_sha256"] for t in flat["tiles"]] == [t["mask_sha256"] for t in var["tiles"]]


def test_roof_table_is_stable_per_building_id() -> None:
    from fraktal.render import ROOF_MATERIALS, roof_table

    ids = ["00000000aaaa1111", "00000001ffff2222", "00000002000033333"[:16]]
    full = roof_table(ids, 0.2)
    again = roof_table(ids[:2], 0.2)
    assert (full[1:3] == again[1:3]).all()
    assert len({tuple(r) for r in full[1:]}) == 3
    assert roof_table(ids, 0.0)[1].tolist() == list(ROOF_MATERIALS[0])


def test_class_gains_vary_by_class_and_scale_with_variation() -> None:
    from fraktal.render import class_gains

    g = class_gains(5, 0.2)
    assert not np.allclose(g[10], g[40])
    assert (class_gains(5, 0.0) == 1).all()
    assert np.abs(class_gains(5, 0.4)[10] - 1).sum() > np.abs(g[10] - 1).sum()


def test_tone_scales_channels_and_defaults_to_neutral() -> None:
    land = np.full((4, 4), 30, dtype="uint8")
    none = np.zeros((4, 4), dtype="uint8")
    neutral = shade(land, none, none)
    toned = shade(land, none, none, tone=(0.5, 1.0, 0.0))
    assert (toned[1] == neutral[1]).all()
    assert (toned[2] == 0).all()
    assert abs(int(toned[0, 0, 0]) - int(neutral[0, 0, 0]) // 2) <= 1


def test_tone_reaches_the_written_image_when_varied(tmp_path: Path, terrain: Path) -> None:
    layer = one_building(CENTER, 10, 5.0)
    base = varied(1)
    dark_blue = RenderConfig(**{**base.__dict__, "tone": (1.0, 1.0, 0.0)})
    out = tmp_path / "t"
    render_region(terrain, layer, CENTER, out, dark_blue)
    with rasterio.open(out / "tiles" / "r00_c00.tif") as src:
        assert (src.read(3) == 0).all()


def test_texture_field_is_deterministic_bounded_and_seeded() -> None:
    from fraktal.render import texture_field

    a = texture_field(1, (80, 80), 0.5)
    assert (a == texture_field(1, (80, 80), 0.5)).all()
    assert not np.allclose(a, texture_field(2, (80, 80), 0.5))
    assert np.abs(a).max() <= 1.0 + 1e-6
    assert a.std() > 0.05


def test_texture_modulates_land_but_barely_water() -> None:
    field = np.tile(np.array([[0.5, -0.5]], dtype="float32"), (4, 2))  # 4x4
    grass = np.full((4, 4), 30, dtype="uint8")
    water = np.full((4, 4), 80, dtype="uint8")
    none = np.zeros((4, 4), dtype="uint8")
    g_spread = np.ptp(shade(grass, none, none, texture=field)[1].astype(int))
    w_spread = np.ptp(shade(water, none, none, texture=field)[1].astype(int))
    assert g_spread > 2 * w_spread > 0
    assert np.ptp(shade(grass, none, none)[1].astype(int)) == 0


def test_texture_leaves_labels_untouched_and_is_deterministic(
    tmp_path: Path, terrain: Path
) -> None:
    layer = one_building(CENTER, 10, 5.0)
    cfg = RenderConfig(**{**varied(4).__dict__, "texture": 0.3})
    a = render_region(terrain, layer, CENTER, tmp_path / "a", cfg)
    b = render_region(terrain, layer, CENTER, tmp_path / "b", cfg)
    flat = render_region(terrain, layer, CENTER, tmp_path / "f", varied(4))
    assert read_hash(tmp_path / "a") == read_hash(tmp_path / "b") != read_hash(tmp_path / "f")
    assert [t["mask_sha256"] for t in a["tiles"]] == [t["mask_sha256"] for t in flat["tiles"]]
    assert a["tiles"] == b["tiles"]


def test_haze_transmittance_is_monotonic_in_optical_depth() -> None:
    from fraktal.render import transmittance

    values = [transmittance(tau, 45.0) for tau in (0.0, 0.05, 0.1, 0.3, 1.0)]
    assert values[0] == 1.0
    assert all(a > b for a, b in zip(values, values[1:], strict=False))
    assert transmittance(0.2, 20.0) < transmittance(0.2, 70.0)  # a low sun crosses more air
    with pytest.raises(ValueError):
        transmittance(-0.1, 45.0)


def test_haze_lifts_dark_pixels_and_flattens_contrast() -> None:
    land = np.full((2, 2), 30, dtype="uint8")
    none = np.zeros((2, 2), dtype="uint8")
    shadowed = np.array([[1, 0], [1, 0]], dtype="uint8")
    clear = shade(land, none, shadowed).astype(int)
    hazy = shade(land, none, shadowed, haze=(0.3, 45.0)).astype(int)
    assert (hazy[:, 0, 0] > clear[:, 0, 0]).all()
    assert np.ptp(hazy[0]) < np.ptp(clear[0])


def test_soft_shadow_mask_gives_intermediate_light() -> None:
    land = np.full((1, 3), 30, dtype="uint8")
    none = np.zeros((1, 3), dtype="uint8")
    soft = np.array([[0.0, 0.5, 1.0]], dtype="float32")
    r = shade(land, none, soft)[0, 0].astype(int)
    assert r[0] > r[1] > r[2]
    assert r[2] == int(round(140 * AMBIENT))


def test_penumbra_and_haze_keep_labels_and_stay_deterministic(
    tmp_path: Path, terrain: Path
) -> None:
    layer = one_building(CENTER, 10, 5.0)
    cfg = RenderConfig(**{**varied(4).__dict__, "haze": 0.2, "penumbra_m": 1.0})
    a = render_region(terrain, layer, CENTER, tmp_path / "a", cfg)
    render_region(terrain, layer, CENTER, tmp_path / "b", cfg)
    flat = render_region(terrain, layer, CENTER, tmp_path / "f", varied(4))
    assert read_hash(tmp_path / "a") == read_hash(tmp_path / "b") != read_hash(tmp_path / "f")
    assert [t["mask_sha256"] for t in a["tiles"]] == [t["mask_sha256"] for t in flat["tiles"]]
    assert a["dropped"] == []


def test_lower_ambient_darkens_shadows_and_contrast_clips_extremes() -> None:
    land = np.full((1, 2), 30, dtype="uint8")
    none = np.zeros((1, 2), dtype="uint8")
    shadow = np.array([[1, 0]], dtype="uint8")
    base = shade(land, none, shadow).astype(int)
    deep = shade(land, none, shadow, ambient=0.1).astype(int)
    assert deep[1, 0, 0] < base[1, 0, 0] and deep[1, 0, 1] == base[1, 0, 1]
    stretched = shade(land, none, shadow, contrast=3.0).astype(int)
    assert stretched.min() == 0 or stretched.max() == 255
    assert stretched[0, 0, 1] - stretched[0, 0, 0] > base[0, 0, 1] - base[0, 0, 0]


def test_bright_roofs_follow_the_building_id_and_the_requested_share() -> None:
    from fraktal.render import BRIGHT_ROOF_GAIN, roof_table

    ids = [f"{i:08x}0000{int(i * 65535 / 99):04x}" for i in range(100)]
    plain = roof_table(ids, 0.0)
    bright = roof_table(ids, 0.0, 0.3)
    ratio = bright[1:, 0] / plain[1:, 0]
    assert 0.25 <= (ratio > 1.01).mean() <= 0.35
    assert ratio.max() == pytest.approx(BRIGHT_ROOF_GAIN)
    assert (roof_table(ids, 0.0, 0.0) == plain).all()


def test_blank_regions_hold_no_labels_and_the_tile_still_validates(
    tmp_path: Path, terrain: Path
) -> None:
    layer = one_building(CENTER, 40, 6.0)
    cfg = RenderConfig(**{**SMALL.__dict__, "blank_prob": 1.0, "blank_max_fraction": 0.5})
    out = tmp_path / "b"
    manifest = render_region(terrain, layer, CENTER, out, cfg)
    assert manifest["dropped"] == [] and manifest["tiles"]
    blanked = 0
    for rec in manifest["tiles"]:
        with rasterio.open(out / "tiles" / f"{rec['tile']}.tif") as img:
            rgb = img.read()
        with rasterio.open(out / "tiles" / f"{rec['tile']}_buildings.tif") as msk:
            mask = msk.read(1)
        dark = (rgb == 0).all(axis=0)
        blanked += int(dark.sum() > 50)
        assert not (mask[dark] > 0).any()
    assert blanked >= 1


def test_extremes_are_deterministic(tmp_path: Path, terrain: Path) -> None:
    layer = one_building(CENTER, 40, 6.0)
    cfg = RenderConfig(
        **{
            **varied(2).__dict__,
            "ambient": 0.2,
            "contrast": 1.3,
            "bright_roofs": 0.5,
            "blank_prob": 0.5,
        }
    )
    a = render_region(terrain, layer, CENTER, tmp_path / "a", cfg)
    b = render_region(terrain, layer, CENTER, tmp_path / "b", cfg)
    assert a["tiles"] == b["tiles"]
    assert read_hash(tmp_path / "a") == read_hash(tmp_path / "b")


def test_roads_are_painted_by_class_and_leave_the_labels_alone(
    tmp_path: Path, terrain: Path
) -> None:
    from pyproj import Transformer
    from shapely.geometry import LineString

    epsg = utm_epsg(*CENTER)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(*CENTER)
    lines = [
        {"geometry": LineString([(x - 60, y + 20), (x + 60, y + 20)]), "road_class": "arterial",
         "width_m": 12.0, "bridge": 0.0, "provenance": "procedural:test"},
        {"geometry": LineString([(x - 60, y - 20), (x + 60, y - 20)]), "road_class": "local",
         "width_m": 6.0, "bridge": 0.0, "provenance": "procedural:test"},
    ]  # fmt: skip
    roads = make_layer("road", "w", lines, f"EPSG:{epsg}")
    buildings = one_building(CENTER, 10.0, 8.0)
    plain = render_region(terrain, buildings, CENTER, tmp_path / "plain", SMALL)
    painted = render_region(terrain, buildings, CENTER, tmp_path / "paint", SMALL, roads=roads)
    assert len(plain["tiles"]) == len(painted["tiles"])
    with rasterio.open(tmp_path / "plain" / "tiles" / "r00_c00.tif") as a:
        flat = a.read()
    with rasterio.open(tmp_path / "paint" / "tiles" / "r00_c00.tif") as b:
        rgb = b.read()
    assert (flat != rgb).any()  # roads changed the picture
    for name in ("buildings", "landcover"):  # labels are identical with and without roads
        dirs = (tmp_path / "plain", tmp_path / "paint")
        files = [t / "tiles" / f"r00_c00_{name}.tif" for t in dirs]
        with rasterio.open(files[0]) as a, rasterio.open(files[1]) as b:
            assert (a.read() == b.read()).all()
    again = render_region(terrain, buildings, CENTER, tmp_path / "again", SMALL, roads=roads)
    with rasterio.open(tmp_path / "again" / "tiles" / "r00_c00.tif") as c:
        assert (c.read() == rgb).all()
    assert again["tiles"]


def test_contrast_stretches_about_the_channel_means_so_dark_channels_survive() -> None:
    land = np.array([[30, 60]], dtype="uint8")
    none = np.zeros((1, 2), dtype="uint8")
    shadow = np.zeros((1, 2), dtype="uint8")
    dark_blue = (1.0, 1.0, 0.25)  # a blue channel far below mid-grey, like the real chips
    plain = shade(land, none, shadow, tone=dark_blue).astype(float)
    stretched = shade(land, none, shadow, tone=dark_blue, contrast=1.5).astype(float)
    assert stretched[2].mean() == pytest.approx(plain[2].mean(), abs=1.0)  # mean kept
    assert abs(np.diff(stretched[2][0])[0]) > abs(np.diff(plain[2][0])[0])  # but spread wider


def test_roof_material_sets_the_roof_colour_family() -> None:
    ids = ["0123456789abcdef", "fedcba9876543210"]  # real ids are 16 hex characters
    table = roof_table(ids, 0.0, materials=["tile", "tin"])
    assert table[1][0] > table[1][2] + 40  # clay tile is red
    assert abs(table[2][0] - table[2][2]) < 60 and table[2][1] > table[1][1]  # tin is greyer


def test_roof_table_accepts_every_generated_building_id(tmp_path: Path) -> None:
    from fraktal import biomes
    from fraktal.procedural import generate_world, read_buildings

    generate_world(
        tmp_path, (104.0, 1.5), 1, biomes.resolve("informal_settlement", {"size_m": 400.0})
    )
    b = read_buildings(tmp_path)
    table = roof_table(b["id"].to_numpy(), 0.1, 0.1, None, b["roof_material"].to_numpy())
    assert table.shape == (len(b) + 1, 3) and (table[1:] > 0).all()
