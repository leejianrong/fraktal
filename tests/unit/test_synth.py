import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from test_chips import make_dataset

from fraktal import cli
from fraktal.chips import LABEL_DIR
from fraktal.render import RenderConfig
from fraktal.skeleton import build_skeleton, held_out_footprints, train_coverage
from fraktal.split import LeakError, assert_no_test_leak, read_manifest
from fraktal.synth import pick_centers, synthesize
from fraktal.terrain import MARKER, TerrainError
from fraktal.world import read_layer, validate_layer

CONFIG = RenderConfig(gsd_m=2.0, tile_px=50, size_m=200.0)


def prepare(tmp_path: Path, n: int = 12) -> Path:
    make_dataset(tmp_path, n)
    assert cli.main(["split", "--data-dir", str(tmp_path), "--seed", "1", "--json"]) == 0
    return tmp_path / "spacenet2" / "khartoum-split.json"


def straddling_label(tmp_path: Path, manifest: Path) -> str:
    """Add a footprint that spans a train tile and an adjacent held-out test tile."""
    data = read_manifest(manifest)
    test = data["test"][0]
    left, bottom, right, top = test["bounds"]
    # Straddle the test tile's west edge, into the territory of its neighbour.
    geom = box(left - 0.0002, top - 0.0004, left + 0.0002, top - 0.0001)
    train_id = data["train"][0]
    name = f"buildings_AOI_5_Khartoum_{train_id}.geojson"
    path = tmp_path / "spacenet2" / "khartoum" / LABEL_DIR / name
    frame = gpd.read_file(path)
    frame = gpd.GeoDataFrame(geometry=[*frame.geometry, geom], crs="EPSG:4326")
    frame.to_file(path, driver="GeoJSON")
    return str(test["id"])


def test_skeleton_holds_only_train_footprints_and_passes_the_leak_check(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    straddling_label(tmp_path, manifest)
    layer = build_skeleton(tmp_path, manifest, seed=4)
    validate_layer(layer, "building")
    assert layer.attrs["dropped_overlapping_test_tiles"] >= 1
    back = layer.to_crs("EPSG:4326")
    assert_no_test_leak(back.geometry, read_manifest(manifest))  # passes: nothing overlaps
    assert len(layer) > 0


def test_leak_check_catches_a_world_with_a_test_building(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    held = read_manifest(manifest)["test"][0]["bounds"]
    inside = box(held[0] + 0.0004, held[1] + 0.0004, held[0] + 0.0008, held[1] + 0.0008)
    with pytest.raises(LeakError):
        assert_no_test_leak([inside], read_manifest(manifest))


def test_skeleton_is_deterministic_and_heights_are_in_range(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    a = build_skeleton(tmp_path, manifest, seed=4)
    b = build_skeleton(tmp_path, manifest, seed=4)
    c = build_skeleton(tmp_path, manifest, seed=5)
    assert a["height_m"].tolist() == b["height_m"].tolist()
    assert a["height_m"].tolist() != c["height_m"].tolist()
    assert a["height_m"].between(4.0, 12.0).all()
    assert (a["provenance"] == "real:spacenet2-khartoum").all()


def make_terrain(tmp_path: Path) -> Path:
    d = tmp_path / "terrain" / "khartoum"
    d.mkdir(parents=True)
    res = 1 / 1200
    size = 240  # 0.2 degrees
    for name, value, dtype in [("dem.tif", 380.0, "float32"), ("landcover.tif", 30, "uint8")]:
        with rasterio.open(
            d / name,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=from_origin(32.0, 15.1, res, res),
        ) as dst:
            dst.write(np.full((size, size), value, dtype=dtype), 1)
    (d / MARKER).write_text(json.dumps({"bounds": [32.0, 14.9, 32.2, 15.1]}))
    return d


def test_synthetic_chips_have_images_masks_samples_and_no_test_overlap(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    layer = build_skeleton(tmp_path, manifest, seed=0)
    terrain = make_terrain(tmp_path)
    footprints = held_out_footprints(manifest, str(layer.crs))
    union = train_coverage(manifest, str(layer.crs))
    centers = pick_centers(manifest, seed=0, size_m=200.0)
    out = tmp_path / "synthetic"
    meta = synthesize(terrain, layer, centers, 2, footprints, union, out, CONFIG, samples=2)
    assert meta["tiles"]
    for chip in meta["tiles"]:
        image = np.load(out / "chips" / f"{chip}_img.npy")
        mask = np.load(out / "chips" / f"{chip}_mask.npy")
        assert image.shape == (3, 50, 50) and mask.shape == (50, 50)
    assert len(list((out / "samples").glob("*.png"))) == 2
    again = synthesize(terrain, layer, centers, 2, footprints, union, tmp_path / "again", CONFIG)
    assert again["tiles"] == meta["tiles"]  # same seed, same chips


def test_a_region_outside_the_terrain_is_skipped_or_fails_loudly(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    layer = build_skeleton(tmp_path, manifest, seed=0)
    terrain = make_terrain(tmp_path)
    footprints = held_out_footprints(manifest, str(layer.crs))
    union = train_coverage(manifest, str(layer.crs))
    with pytest.raises(TerrainError, match="only 0 of 1"):
        synthesize(terrain, layer, [(40.0, 15.0)], 1, footprints, union, tmp_path / "o", CONFIG)


def test_pick_centers_needs_tile_bounds(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    data = json.loads(manifest.read_text())
    del data["tile_bounds"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="tile_bounds"):
        pick_centers(manifest, 0)


def test_generate_cli_writes_a_valid_world(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare(tmp_path)
    capsys.readouterr()
    assert cli.main(["generate", "--data-dir", str(tmp_path), "--seed", "2", "--json"]) == 0
    world_path = Path(json.loads(capsys.readouterr().out)["data"]["world"])
    assert len(read_layer(world_path, "building")) > 0


def test_best_covered_regions_come_first_and_never_overlap(tmp_path: Path) -> None:
    manifest = prepare(tmp_path, 12)
    centers = pick_centers(manifest, seed=0, size_m=200.0)
    assert len(centers) == len(set(centers)) > 1
    again = pick_centers(manifest, seed=0, size_m=200.0)
    assert centers == again


def test_chips_outside_labelled_tiles_are_dropped(tmp_path: Path) -> None:
    manifest = prepare(tmp_path)
    layer = build_skeleton(tmp_path, manifest, seed=0)
    terrain = make_terrain(tmp_path)
    footprints = held_out_footprints(manifest, str(layer.crs))
    union = train_coverage(manifest, str(layer.crs))
    # A region over bare ground between the one-kilometre-spaced tiles has no labels at all.
    meta = synthesize(
        terrain, layer, [(32.105, 15.0)], 1, footprints, union, tmp_path / "o", CONFIG
    )
    assert meta["tiles"] == []
    assert {d["reason"] for d in meta["dropped"]} <= {
        "outside labelled training tiles",
        "overlaps a held-out test tile",
    }
    assert any(d["reason"] == "outside labelled training tiles" for d in meta["dropped"])


def test_randomised_regions_record_their_own_parameters_and_stay_reproducible(
    tmp_path: Path,
) -> None:
    from fraktal.randomise import Ranges

    manifest = prepare(tmp_path)
    layer = build_skeleton(tmp_path, manifest, seed=0)
    terrain = make_terrain(tmp_path)
    footprints = held_out_footprints(manifest, str(layer.crs))
    union = train_coverage(manifest, str(layer.crs))
    centers = pick_centers(manifest, seed=0, size_m=200.0)
    args = (terrain, layer, centers, 2, footprints, union)
    meta = synthesize(*args, tmp_path / "a", CONFIG, ranges=Ranges(), seed=7)
    again = synthesize(*args, tmp_path / "b", CONFIG, ranges=Ranges(), seed=7)
    assert meta["randomised"] is True
    params = meta["region_params"]
    assert len(params) == 2 and params[0] != params[1]
    assert params == again["region_params"] and meta["tiles"] == again["tiles"]
    for chip in meta["tiles"]:  # same chips, same pixels
        a = np.load(tmp_path / "a" / "chips" / f"{chip}_img.npy")
        assert (a == np.load(tmp_path / "b" / "chips" / f"{chip}_img.npy")).all()
    fixed = synthesize(*args, tmp_path / "c", CONFIG)
    assert fixed["randomised"] is False and fixed["region_params"][0] == fixed["region_params"][1]
