import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from fraktal import refboard
from fraktal.chips import tile_path

GAINS = (1.0, 1.15, 0.75)  # a raw band imbalance like the Khartoum one: blue low, green high


def _write_tile(root: Path, tile_id: str, seed: int) -> None:
    path = tile_path(root, tile_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.random.default_rng(seed).uniform(300, 900, (32, 32))
    data = np.stack([base * g for g in GAINS]).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", width=32, height=32, count=3, dtype="uint16",
        crs="EPSG:4326", transform=from_origin(0, 0, 1e-5, 1e-5),
    ) as dst:  # fmt: skip
        dst.write(data)


def test_a_per_band_view_removes_the_cast_a_shared_stretch_keeps(tmp_path: Path) -> None:
    ids = ["img1", "img2"]
    for k, tile_id in enumerate(ids):
        _write_tile(tmp_path, tile_id, k)
    ranges = refboard.band_ranges(tmp_path, ids)
    assert ranges.shape == (3, 2) and (ranges[:, 1] > ranges[:, 0]).all()
    assert ranges[2, 1] < ranges[1, 1]  # the raw blue band is lower than green
    view = refboard.spacenet_view(tmp_path, "img1", ranges)
    assert view.shape == (3, 32, 32) and view.dtype == np.uint8
    means = view.reshape(3, -1).mean(axis=1)
    assert means.max() - means.min() < 8  # balanced
    raw = np.stack([np.random.default_rng(0).uniform(300, 900, (32, 32)) * g for g in GAINS])
    shared = np.clip((raw - 300) / 600 * 255, 0, 255).reshape(3, -1).mean(axis=1)
    assert shared.max() - shared.min() > 40  # one shared range leaves the imbalance


def test_no_data_is_ignored_in_the_ranges(tmp_path: Path) -> None:
    _write_tile(tmp_path, "img1", 0)
    with rasterio.open(tile_path(tmp_path, "img1"), "r+") as dst:
        data = dst.read()
        data[:, :8, :] = 0
        dst.write(data)
    assert refboard.band_ranges(tmp_path, ["img1"])[:, 0].min() > 200


def _fake_data(data: Path) -> Path:
    """Reference chips for two biomes and renders for them, all small."""
    pytest.importorskip("PIL")
    from PIL import Image

    maxar = data / "eval" / "maxar-reference"
    maxar.mkdir(parents=True)
    for event in ("Typhoon-Kalmaegi-Nov-2025", "Kenya-Flooding-May24"):
        name = f"{event}__47_000_2025-01-01_abc__busy"
        chip = np.random.default_rng(1).integers(60, 200, (3, 64, 64)).astype("uint8")
        chip[:, :, :32] //= 3  # large-scale structure, so the chip counts as usable
        np.save(maxar / f"{name}.npy", chip)
        meta = {"event": event, "scene": "47/000/2025-01-01/abc-visual.tif", "kind": "busy"}
        (maxar / f"{name}.json").write_text(json.dumps({**meta, "pixel_size": [0.31, 0.31]}))
    synthetic = data / "synth"
    for biome in ("dense_urban", "mangrove_coast"):
        tile = synthetic / biome / "tile_0000"
        tile.mkdir(parents=True)
        Image.fromarray(np.full((100, 100, 3), 200, dtype="uint8")).save(tile / "sensor.png")
    return synthetic


def test_review_writes_a_sheet_with_a_row_per_biome_and_real_chips_on_each(tmp_path: Path) -> None:
    from fraktal.cli import main

    data = tmp_path / "data"
    synthetic = _fake_data(data)
    from PIL import Image

    args = ["review", "--data-dir", str(data), "--synthetic", str(synthetic), "--per-row", "2"]
    assert main([*args, "--board"]) == 0
    sheet = Image.open(data / "review" / "sheet_satellite.jpg")
    assert sheet.size == (3 * 300, 2 * (300 + 18))  # one synthetic and two real chips per row
    assert "never train on" in (data / "review" / "reference-board.html").read_text()


def test_review_refuses_to_write_real_imagery_outside_the_data_dir(tmp_path: Path) -> None:
    from fraktal.cli import main

    data = tmp_path / "data"
    synthetic = _fake_data(data)
    args = ["review", "--data-dir", str(data), "--synthetic", str(synthetic)]
    assert main([*args, "--out", str(tmp_path / "elsewhere")]) == 2
    assert not (tmp_path / "elsewhere").exists()


def test_unusable_chips_are_cloud_no_data_or_featureless() -> None:
    rng = np.random.default_rng(0)
    good = rng.integers(40, 180, (64, 64, 3)).astype("uint8")
    good[:, :32] //= 3  # large-scale structure: half the chip is darker
    cloud = np.full((64, 64, 3), 240, dtype="uint8")
    edge = good.copy()
    edge[:, :16] = 0
    sea = np.full((64, 64, 3), 90, dtype="uint8") + rng.integers(0, 6, (64, 64, 3)).astype("uint8")
    assert refboard.usable(good)
    assert not refboard.usable(cloud) and not refboard.usable(edge) and not refboard.usable(sea)


def test_ground_photos_are_listed_per_place_and_source_and_shown_on_the_board(
    tmp_path: Path,
) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    root = tmp_path / "eval" / "ground-reference"
    root.mkdir(parents=True)
    for name in ("mapillary__kampala__1", "mapillary__kampala__2", "mapillary__kampala__3",
                 "commons__kampala__a", "mapillary__cebu__9"):  # fmt: skip
        Image.fromarray(np.full((30, 40, 3), 100, dtype="uint8")).save(root / f"{name}.jpg")
    chips = refboard.ground_chips(tmp_path, per_place=2)
    assert [c.label for c in chips] == [
        "commons kampala",
        "mapillary cebu",
        "mapillary kampala",
        "mapillary kampala",
    ]
    assert chips[0].image.shape == (30, 40, 3)  # a photo keeps its aspect
    board = refboard.write_board(tmp_path, tmp_path / "out")
    assert "Ground level" in board.read_text()


def _pull_maxar() -> Any:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "refboard" / "pull_maxar.py"
    spec = importlib.util.spec_from_file_location("pull_maxar", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_maxar_planner_prefers_before_and_after_scenes_of_the_same_tile() -> None:
    pull = _pull_maxar()
    event = "Earthquake-Myanmar-March-2025"  # 2025-03-28
    keys = [
        (f"events/{event}/ard/47/A/2025-02-15/x-visual.tif", 100),
        (f"events/{event}/ard/47/A/2025-04-04/y-visual.tif", 90),
        (f"events/{event}/ard/47/B/2025-04-04/z-visual.tif", 500),  # big, but only after the event
        (f"events/{event}/ard/47/C/2025-01-01/p-visual.tif", 80),
        (f"events/{event}/ard/47/C/2025-03-28/q-visual.tif", 70),  # on the day counts as after
        (f"events/{event}/ard/47/D/2025-01-02/r-visual.tif", 60),
    ]

    def k(tile: str, date: str, name: str) -> str:
        return f"events/{event}/ard/47/{tile}/{date}/{name}-visual.tif"

    plan = pull.plan(event, keys)
    assert plan == [
        (k("A", "2025-02-15", "x"), k("A", "2025-04-04", "y")),
        (k("C", "2025-01-01", "p"), k("C", "2025-03-28", "q")),
    ]
    only_single = pull.plan(event, keys[2:3] + keys[5:])
    assert only_single == [(None, keys[2][0]), (keys[5][0], None)]
