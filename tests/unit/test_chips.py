import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from fraktal import cli
from fraktal.chips import LABEL_DIR, ChipTiles, build_chips, load_splits
from fraktal.split import IMAGERY_DIR


def make_dataset(data_dir: Path, n: int, size: int = 20) -> None:
    root = data_dir / "spacenet2" / "khartoum"
    (root / IMAGERY_DIR).mkdir(parents=True)
    (root / LABEL_DIR).mkdir(parents=True)
    for i in range(1, n + 1):
        left = 32.0 + i * 0.01
        transform = from_origin(left, 15.0, 0.0001, 0.0001)
        data = np.full((3, size, size), 500, dtype="uint16")
        data[:, 5:10, 5:10] = 3000  # a bright "roof"
        with rasterio.open(
            root / IMAGERY_DIR / f"RGB-PanSharpen_AOI_5_Khartoum_img{i}.tif",
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=3,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(data)
        if i != 2:  # tile img2 has no label file at all
            roof = box(left + 5 * 0.0001, 15.0 - 10 * 0.0001, left + 10 * 0.0001, 15.0 - 5 * 0.0001)
            gpd.GeoDataFrame(geometry=[roof], crs="EPSG:4326").to_file(
                root / LABEL_DIR / f"buildings_AOI_5_Khartoum_img{i}.geojson", driver="GeoJSON"
            )


def split_and_build(tmp_path: Path, n: int = 10) -> Path:
    make_dataset(tmp_path, n)
    assert cli.main(["split", "--data-dir", str(tmp_path), "--seed", "1", "--json"]) == 0
    return build_chips(tmp_path, tmp_path / "spacenet2" / "khartoum-split.json")


def test_chips_pair_each_image_with_its_mask(tmp_path: Path) -> None:
    out = split_and_build(tmp_path)
    train, test = load_splits(out)
    assert len(train) == 8 and len(test) == 2
    for source in (train, test):
        for i in range(len(source)):
            image, mask = source.get(i)
            assert image.dtype == np.uint8 and image.shape == (3, 20, 20)
            assert mask.shape == (20, 20)
            if source.ids[i] == "img2":
                assert mask.sum() == 0  # no labels means no buildings
            else:
                assert mask.sum() == 25
                assert image[:, mask > 0].mean() > image[:, mask == 0].mean()


def test_build_is_idempotent_for_the_same_split(tmp_path: Path) -> None:
    out = split_and_build(tmp_path)
    stamp = (out / "chips.json").stat().st_mtime_ns
    build_chips(tmp_path, tmp_path / "spacenet2" / "khartoum-split.json")
    assert (out / "chips.json").stat().st_mtime_ns == stamp


def test_changing_the_seed_rebuilds(tmp_path: Path) -> None:
    out = split_and_build(tmp_path)
    args = ["split", "--data-dir", str(tmp_path), "--seed", "2", "--json"]
    assert cli.main(args) == 0
    build_chips(tmp_path, tmp_path / "spacenet2" / "khartoum-split.json")
    after = json.loads((out / "chips.json").read_text())
    assert after["seed"] == 2


def test_tile_source_sorts_ids(tmp_path: Path) -> None:
    out = split_and_build(tmp_path)
    assert ChipTiles(out, ["img3", "img1"]).ids == ["img1", "img3"]


def test_benchmark_cli_runs_end_to_end_on_cpu(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    make_dataset(tmp_path, 12, size=300)
    assert cli.main(["split", "--data-dir", str(tmp_path), "--seed", "1", "--json"]) == 0
    capsys.readouterr()
    args = ["benchmark", "--data-dir", str(tmp_path), "--steps", "2", "--device", "cpu", "--json"]
    assert cli.main(args) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert set(envelope["data"]["iou"]) == {"real_10pct", "real_100pct"}
    report = json.loads(Path(envelope["data"]["report"]).read_text())
    assert report["recipe"]["steps"] == 2 and report["seed"] == 0
