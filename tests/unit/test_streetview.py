from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio

from fraktal import streetview


def _street() -> np.ndarray[Any, Any]:
    """Blue sky over a grey road with a green hedge and a dark wall edge."""
    img = np.zeros((100, 200, 3), dtype="uint8")
    img[:50] = (120, 170, 230)  # sky
    img[50:] = (110, 110, 110)  # road
    img[50:70, :60] = (40, 150, 40)  # hedge
    img[50:, 150:] = (30, 30, 30)  # a wall: a vertical edge at x=150
    return img


def _write(path: Path, img: np.ndarray[Any, Any]) -> None:
    with rasterio.open(
        path, "w", driver="PNG", height=img.shape[0], width=img.shape[1], count=3, dtype="uint8"
    ) as dst:
        dst.write(np.moveaxis(img, -1, 0))


def test_statistics_see_sky_vegetation_and_vertical_edges() -> None:
    s = streetview.statistics(_street())
    assert s["sky_share"] == pytest.approx(1.0)  # the whole top half is sky
    assert 0.05 < s["green_share"] < 0.15
    assert s["vertical_structure"] < 0.5  # the long sky/road horizon outweighs the wall edge
    wall = np.zeros((40, 40, 3), dtype="uint8")
    wall[:, 20:] = 200
    assert streetview.statistics(wall)["vertical_structure"] == 1.0
    assert set(s) == set(streetview.STATS)
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_a_dark_image_has_no_sky_and_low_brightness() -> None:
    s = streetview.statistics(np.full((60, 60, 3), 20, dtype="uint8"))
    assert s["sky_share"] == 0.0 and s["brightness"] < 0.1 and s["edge_energy"] == 0.0


def test_wasserstein_is_zero_for_the_same_sample_and_grows_with_the_shift() -> None:
    a = [0.1, 0.2, 0.3, 0.4]
    assert streetview.wasserstein(a, a) == 0.0
    assert streetview.wasserstein(a, [x + 0.5 for x in a]) == pytest.approx(0.5)


def test_compare_flags_a_synthetic_set_that_is_far_from_the_photos(tmp_path: Path) -> None:
    real = [streetview.Photo(tmp_path, "commons", "x", _street() + k) for k in range(0, 4)]
    same = [streetview.Photo(tmp_path, "synthetic", "x", _street() + k) for k in range(0, 4)]
    dark = [streetview.Photo(tmp_path, "synthetic", "x", np.full((100, 200, 3), 15, "uint8"))]
    ok = streetview.compare(real, same)
    assert ok["low_n"] and ok["n_photos"] == 4 and not any(r["flag"] for r in ok["stats"].values())
    far = streetview.compare(real, dark)
    assert far["stats"]["sky_share"]["flag"] and far["stats"]["brightness"]["z"] < -2
    assert "*" in streetview.table(far)
    with pytest.raises(ValueError, match="at least one"):
        streetview.compare([], same)


def test_loaders_and_the_sheet_work_on_files(tmp_path: Path) -> None:
    ref = tmp_path / "ref"
    ref.mkdir()
    for name in ("commons__nairobi-centre__a.jpg", "mapillary__nairobi-mathare__b.jpg"):
        with rasterio.open(
            ref / name, "w", driver="JPEG", height=100, width=200, count=3, dtype="uint8"
        ) as dst:
            dst.write(np.moveaxis(_street(), -1, 0))
    assert len(streetview.load_reference(ref, "*")) == 2
    only = streetview.load_reference(ref, "nairobi-c*")
    assert [p.place for p in only] == ["nairobi-centre"]
    frames = tmp_path / "frames"
    frames.mkdir()
    for k in range(5):
        _write(frames / f"frame_{k:04d}.png", _street())
    synth = streetview.load_frames(frames, limit=3)
    assert [p.path.name for p in synth] == ["frame_0000.png", "frame_0002.png", "frame_0004.png"]
    out = tmp_path / "sheet.png"
    streetview.sheet(only, synth, out, cell=64)
    with rasterio.open(out) as src:
        assert src.width == 6 * 64 and src.height == 2 * 64  # one row of frames, one of photos
    assert out.with_suffix(".json").exists()
