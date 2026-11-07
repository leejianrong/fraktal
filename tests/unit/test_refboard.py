from pathlib import Path

import numpy as np
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
