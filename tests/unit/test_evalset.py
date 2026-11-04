import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from fraktal import chips, cli
from fraktal.evalset import EvalSetError, build_eval_set, parse_scene

RES = 0.1  # metres per source pixel; chips at 0.5 m take 5 x 5 source pixels per output pixel


def _scene(root: Path, city: str, sid: str) -> None:
    """A 200 x 100 m scene, RGBA, with two buildings, in the Open Cities folder layout."""
    w, h = 2000, 1000
    folder = root / "train_tier_1" / city / sid
    folder.mkdir(parents=True)
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    rgba[0], rgba[1], rgba[2], rgba[3] = 90, 120, 60, 255
    rgba[:3, 300:500, 300:700] = 200  # a bright roof, 20 x 40 m
    rgba[3, :, 1900:] = 0  # no-data strip on the right edge
    transform = from_origin(500000.0, 4000100.0, RES, RES)
    with rasterio.open(
        folder / f"{sid}.tif", "w", driver="GTiff", width=w, height=h, count=4,
        dtype="uint8", crs="EPSG:32636", transform=transform,
    ) as dst:  # fmt: skip
        dst.write(rgba)
    labels = root / "train_tier_1" / city / f"{sid}-labels"
    labels.mkdir()
    roof = box(500030.0, 4000050.0, 500070.0, 4000070.0)  # matches the bright block
    other = box(500120.0, 4000020.0, 500140.0, 4000040.0)
    gpd.GeoDataFrame(geometry=[roof, other], crs="EPSG:32636").to_crs("EPSG:4326").to_file(
        labels / f"{sid}.geojson", driver="GeoJSON"
    )


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "oc"
    _scene(path, "dar", "abc123")
    return path


def test_parse_scene() -> None:
    assert parse_scene("dar/a017f9").name == "dar-a017f9"
    for bad in ("dar", "dar/", "/x", "a/b/c"):
        with pytest.raises(EvalSetError):
            parse_scene(bad)


def test_chips_have_the_chip_format_labels_on_the_grid_and_an_eval_only_flag(
    root: Path, tmp_path: Path
) -> None:
    out = build_eval_set(
        [parse_scene("dar/abc123")], tmp_path / "out", base=str(root),
        gsd_m=0.5, chip_px=40, max_chips_per_scene=50, seed=0,
    )  # fmt: skip
    meta = json.loads((out / "chips.json").read_text())
    assert meta["eval_only"] is True and meta["train"] == [] and meta["licence"] == "ODbL-1.0"
    assert (out / "EVAL_ONLY.txt").is_file()
    _, test = chips.load_splits(out)
    assert len(test) == len(meta["tiles"]) > 3
    image, mask = test.get(0)
    assert image.shape == (3, 40, 40) and image.dtype == np.uint8 and mask.shape == (40, 40)
    # the labelled roof is bright in the image: mask and imagery agree where both exist
    inside = np.zeros(len(test), dtype=float)
    for i in range(len(test)):
        img, msk = test.get(i)
        if msk.sum() and (msk == 0).any():
            inside[i] = float(img[:, msk > 0].mean() - img[:, msk == 0].mean())
    assert inside.max() > 40
    # nothing is taken from the no-data strip: chips are 20 m wide, the strip starts at 190 m
    assert max(int(name.rsplit("c", 1)[1]) for name in meta["tiles"]) <= 8


def test_empty_chips_are_capped_and_resolution_is_checked(root: Path, tmp_path: Path) -> None:
    out = build_eval_set(
        [parse_scene("dar/abc123")], tmp_path / "o2", base=str(root),
        gsd_m=0.5, chip_px=40, max_chips_per_scene=10, seed=1,
    )  # fmt: skip
    _, test = chips.load_splits(out)
    empty = sum(int(test.get(i)[1].sum() == 0) for i in range(len(test)))
    assert empty <= 1
    with pytest.raises(EvalSetError, match="coarser"):
        build_eval_set([parse_scene("dar/abc123")], tmp_path / "o3", base=str(root), gsd_m=0.05)


def test_same_seed_same_chips(root: Path, tmp_path: Path) -> None:
    a = build_eval_set([parse_scene("dar/abc123")], tmp_path / "a", base=str(root), chip_px=40)
    b = build_eval_set([parse_scene("dar/abc123")], tmp_path / "b", base=str(root), chip_px=40)
    assert (a / "chips.json").read_text() == (b / "chips.json").read_text()
    first = sorted(p.name for p in a.glob("*_img.npy"))[0]
    assert (np.load(a / first) == np.load(b / first)).all()


def test_eval_set_cli(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cli"
    code = cli.main(
        ["eval-set", "dar/abc123", "--base", str(root), "--chip-px", "40",
         "--max-chips", "6", "--out", str(out), "--json"]
    )  # fmt: skip
    data = json.loads(capsys.readouterr().out)
    assert code == 0 and data["data"]["tiles"] >= 3
    assert cli.main(["eval-set", "bad", "--out", str(tmp_path / "x")]) == 4
