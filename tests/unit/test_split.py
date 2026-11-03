import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from fraktal import cli
from fraktal.split import (
    IMAGERY_DIR,
    LeakError,
    Tile,
    assert_no_test_leak,
    build_manifest,
    list_tiles,
    read_manifest,
    split_tiles,
    write_manifest,
)

TILES = [Tile(f"img{i}", (float(i), 0.0, float(i) + 1.0, 1.0)) for i in range(1, 51)]


def test_same_seed_gives_same_split() -> None:
    assert split_tiles(TILES, 7, 0.2) == split_tiles(TILES, 7, 0.2)
    assert split_tiles(TILES, 7, 0.2) != split_tiles(TILES, 8, 0.2)


def test_split_ignores_input_order() -> None:
    assert split_tiles(TILES, 3, 0.2) == split_tiles(list(reversed(TILES)), 3, 0.2)


def test_no_tile_in_both_sets_and_none_lost() -> None:
    train, test = split_tiles(TILES, 1, 0.2)
    assert not {t.id for t in train} & {t.id for t in test}
    assert len(train) + len(test) == len(TILES)
    assert len(test) == 10


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_bad_fraction_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        split_tiles(TILES, 0, fraction)


def test_tiny_fraction_still_holds_out_one_tile() -> None:
    _, test = split_tiles(TILES[:3], 0, 0.01)
    assert len(test) == 1


def test_manifest_records_sizes_and_seed(tmp_path: Path) -> None:
    manifest = build_manifest(TILES, "EPSG:4326", 42, 0.2)
    assert manifest["seed"] == 42
    assert manifest["counts"] == {"train": 40, "test": 10, "total": 50}
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    assert read_manifest(path) == manifest


def test_manifest_schema_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema": 99}))
    with pytest.raises(ValueError, match="schema"):
        read_manifest(path)


def test_leak_check_fails_loudly_on_overlap() -> None:
    manifest = build_manifest(TILES, "EPSG:4326", 0, 0.2)
    held_out = manifest["test"][0]
    inside = Point(
        (held_out["bounds"][0] + held_out["bounds"][2]) / 2,
        (held_out["bounds"][1] + held_out["bounds"][3]) / 2,
    )
    with pytest.raises(LeakError, match=held_out["id"]):
        assert_no_test_leak([inside], manifest)


def test_leak_check_passes_for_train_only_geometry_and_edge_touch() -> None:
    manifest = build_manifest(TILES, "EPSG:4326", 0, 0.2)
    test_ids = {t["id"] for t in manifest["test"]}
    train_boxes = [box(*t.bounds) for t in TILES if t.id not in test_ids]
    # Tiles here share edges only; touching must not count as a leak.
    assert_no_test_leak(train_boxes, manifest)


def _write_tif(path: Path, left: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(left, 1.0, 0.25, 0.25),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype="uint8"))


def make_extracted(root: Path, n: int) -> None:
    imagery = root / IMAGERY_DIR
    imagery.mkdir(parents=True)
    for i in range(1, n + 1):
        _write_tif(imagery / f"RGB-PanSharpen_AOI_5_Khartoum_img{i}.tif", float(i))


def test_list_tiles_reads_ids_and_bounds(tmp_path: Path) -> None:
    make_extracted(tmp_path, 3)
    tiles, crs = list_tiles(tmp_path)
    assert [t.id for t in tiles] == ["img1", "img2", "img3"]
    assert tiles[0].bounds == (1.0, 0.0, 2.0, 1.0)
    assert crs == "EPSG:4326"


def test_list_tiles_without_data_points_at_ingest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingest"):
        list_tiles(tmp_path)


def test_cli_split_writes_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_extracted(tmp_path / "spacenet2" / "khartoum", 10)
    args = ["split", "--data-dir", str(tmp_path), "--seed", "5", "--json"]
    assert cli.main(args) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["data"]["counts"] == {"train": 8, "test": 2, "total": 10}
    manifest = read_manifest(Path(envelope["data"]["manifest"]))
    assert manifest["seed"] == 5


def test_cli_split_without_data_fails_with_exit_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["split", "--data-dir", str(tmp_path), "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["error"]["kind"] == "split_failed"
