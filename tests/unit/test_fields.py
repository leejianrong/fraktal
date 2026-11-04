import math
from pathlib import Path

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import box

from fraktal import biomes
from fraktal.fields import access_tracks, row_points
from fraktal.procedural import generate_world, read_buildings, read_fields, read_roads
from fraktal.render import RenderConfig, render_region

CENTER = (101.5, 3.0)


def _edge_angles(poly: object) -> set[float]:
    pts = list(poly.exterior.coords)  # type: ignore[attr-defined]
    return {
        round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0, 3)
        for a, b in zip(pts[:-1], pts[1:], strict=True)
    }


def test_rows_are_parallel_equally_spaced_and_align_with_the_boundary() -> None:
    poly = affinity.rotate(box(0, 0, 300, 200), 27.0, origin=(0, 0))
    pts = row_points(poly, 27.0, 8.5)
    theta = math.radians(27.0)
    across = -pts[:, 0] * math.sin(theta) + pts[:, 1] * math.cos(theta)  # distance across rows
    rows = np.unique(np.round(across, 3))
    gaps = np.diff(rows)
    assert len(rows) > 10
    assert abs(gaps.min() - 0.87 * 8.5) < 0.01
    steps = gaps / gaps.min()  # every gap is one row apart, or two where the track row is left
    assert np.allclose(steps, np.round(steps), atol=0.01) and steps.max() <= 2.01
    along = np.unique(np.round(pts[:, 0] * math.cos(theta) + pts[:, 1] * math.sin(theta), 3))
    assert len(along) > 10
    assert any(abs(a - 27.0) < 0.01 for a in _edge_angles(poly))  # rows run along a side
    assert poly.contains(world_points(pts))


def world_points(pts: np.ndarray) -> object:
    from shapely.geometry import MultiPoint

    return MultiPoint(pts.tolist())


def test_the_centre_row_is_left_for_the_track() -> None:
    poly = box(0, 0, 200, 100)
    with_track = row_points(poly, 0.0, 8.5)
    without = row_points(poly, 0.0, 8.5, leave_track_row=False)
    assert len(without) > len(with_track)


@pytest.fixture(scope="module")
def plantation(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:  # type: ignore[type-arg]
    path = tmp_path_factory.mktemp("palm")
    config = biomes.resolve("oil_palm_plantation", {"size_m": 1200.0})
    return path, generate_world(path, CENTER, 2, config)


def test_plantation_fields_do_not_overlap_or_cover_buildings_roads_or_water(
    plantation: tuple[Path, dict],  # type: ignore[type-arg]
) -> None:
    path, manifest = plantation
    fields = read_fields(path)
    assert fields is not None and len(fields) >= 3
    geoms = list(fields.geometry)
    for i, a in enumerate(geoms):
        assert all(not a.intersects(b) for b in geoms[i + 1 :])
    buildings = read_buildings(path)
    assert not any(g.intersects(b) for g in geoms for b in buildings.geometry)
    arterials = read_roads(path)
    arterials = arterials[arterials["road_class"] != "track"]
    assert not any(g.intersects(r) for g in geoms for r in arterials.geometry)
    assert set(fields["field_type"]) == {"oil_palm"}
    for _, f in fields.iterrows():
        assert any(abs(a - f.row_angle_deg) < 0.01 for a in _edge_angles(f.geometry))
    assert "field" in manifest["layers"]


def test_tracks_lie_inside_their_fields_and_are_written_as_roads(
    plantation: tuple[Path, dict],  # type: ignore[type-arg]
) -> None:
    path, _ = plantation
    fields = read_fields(path)
    tracks = read_roads(path)
    tracks = tracks[tracks["road_class"] == "track"]
    assert len(tracks) == len(access_tracks(fields, 4.0)) and len(tracks) > 0
    union = fields.geometry.union_all().buffer(0.01)
    assert all(union.contains(t) for t in tracks.geometry)
    assert (tracks["width_m"] == 4.0).all()


def test_plantation_is_byte_identical_per_seed(tmp_path: Path) -> None:
    config = biomes.resolve("oil_palm_plantation", {"size_m": 1200.0})
    one = generate_world(tmp_path / "a", CENTER, 4, config)
    two = generate_world(tmp_path / "b", CENTER, 4, config)
    assert one["layers"] == two["layers"] and one["world_id"] == two["world_id"]


def test_default_worlds_have_no_field_layer(tmp_path: Path) -> None:
    generate_world(tmp_path, CENTER, 1, biomes.resolve("khartoum", {"size_m": 600.0}))
    assert read_fields(tmp_path) is None


def test_rendering_fields_paints_ground_and_trees_but_leaves_labels_alone(
    plantation: tuple[Path, dict],  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    import rasterio

    path, _ = plantation
    cfg = RenderConfig(gsd_m=1.0, tile_px=300, size_m=300.0, objects=1.0, seed=1)
    bare = render_region(path, read_buildings(path), CENTER, tmp_path / "bare", cfg,
                         roads=read_roads(path))  # fmt: skip
    full = render_region(path, read_buildings(path), CENTER, tmp_path / "full", cfg,
                         roads=read_roads(path), fields=read_fields(path))  # fmt: skip
    assert [t["mask_sha256"] for t in bare["tiles"]] == [t["mask_sha256"] for t in full["tiles"]]
    name = full["tiles"][0]["tile"]
    with rasterio.open(tmp_path / "bare" / "tiles" / f"{name}_landcover.tif") as a:
        land_bare = a.read()
    with rasterio.open(tmp_path / "full" / "tiles" / f"{name}_landcover.tif") as b:
        land_full = b.read()
    assert (land_bare == land_full).all()  # land-cover labels never see the fields
    with rasterio.open(tmp_path / "bare" / "tiles" / f"{name}.tif") as a:
        img_bare = a.read()
    with rasterio.open(tmp_path / "full" / "tiles" / f"{name}.tif") as b:
        img_full = b.read()
    assert (img_bare != img_full).any()


@pytest.fixture(scope="module")
def rice(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("rice")
    generate_world(path, CENTER, 2, biomes.resolve("paddy_terraces"))
    return path


def test_paddy_blocks_share_one_grid_and_never_overlap(rice: Path) -> None:
    fields = read_fields(rice)
    assert fields is not None
    paddy = fields[fields["field_type"] == "paddy"]
    assert len(paddy) > 20
    assert paddy["row_angle_deg"].nunique() == 1  # one rotated grid, so blocks tile edge to edge
    geoms = list(paddy.geometry)
    overlap = max(
        (a.intersection(b).area for i, a in enumerate(geoms) for b in geoms[i + 1 :]), default=0.0
    )
    assert overlap < 1e-6
    assert any(a.distance(b) < 0.01 for i, a in enumerate(geoms) for b in geoms[i + 1 :])


def test_paddy_states_are_all_present_and_render_distinct_colours(rice: Path) -> None:
    from fraktal.render import LANDCOVER_RGB, PADDY_CODES

    fields = read_fields(rice)
    assert fields is not None
    assert set(fields[fields["field_type"] == "paddy"]["state"]) == {
        "flooded",
        "planted",
        "harvested",
    }
    colours = {tuple(LANDCOVER_RGB[c]) for c in PADDY_CODES.values()}
    assert len(colours) == 3
    pairs = [(a, b) for a in colours for b in colours if a < b]
    assert all(math.dist(a, b) > 40 for a, b in pairs)


def test_paddy_blocks_are_levelled_in_the_dem(rice: Path) -> None:
    import rasterio
    from rasterio.features import rasterize

    fields = read_fields(rice)
    assert fields is not None
    with rasterio.open(rice / "dem.tif") as src:
        dem, transform = src.read(1), src.transform
    for _, f in fields[fields["field_type"] == "paddy"].head(5).iterrows():
        inside = rasterize([(f.geometry, 1)], dem.shape, transform=transform, dtype="uint8") > 0
        if inside.any():
            assert float(np.ptp(dem[inside])) < 1e-3


def test_terraces_follow_the_contours() -> None:
    from fraktal.render import TERRACE_A, TERRACE_B, TERRACE_RISER, terrace_bands

    ramp = np.tile(np.linspace(0.0, 30.0, 600), (20, 1))  # a slope rising along the columns
    codes = terrace_bands(ramp, 3.0)
    for row in codes:  # a step's colour only changes where the height crosses a multiple of 3 m
        change = np.nonzero(np.diff(row.astype(int)))[0]
        near = np.abs(ramp[0, change + 1] / 3.0 - np.round(ramp[0, change + 1] / 3.0))
        assert (near < 0.12).sum() >= len(change) // 2
    assert {TERRACE_A, TERRACE_B, TERRACE_RISER} <= set(np.unique(codes).tolist())


def test_rice_world_has_terraces_on_slopes_and_is_byte_identical(
    tmp_path: Path, rice: Path
) -> None:
    fields = read_fields(rice)
    assert fields is not None and (fields["field_type"] == "terrace").any()
    again = generate_world(tmp_path, CENTER, 2, biomes.resolve("paddy_terraces"))
    import json

    first = json.loads((rice / "world.json").read_text())
    assert first["layers"] == again["layers"] and first["rasters"] == again["rasters"]
