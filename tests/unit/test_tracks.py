from pathlib import Path

import numpy as np
import pytest
import rasterio

from fraktal import agents, labels3d, tracks


def _write(path: Path, semantic: np.ndarray, instance: np.ndarray) -> None:
    rgb = labels3d.encode(semantic, instance)
    profile = {"driver": "PNG", "count": 3, "dtype": "uint8", "height": rgb.shape[0],
               "width": rgb.shape[1]}  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))


def _tracks(n: int = 3, frames: int = 2) -> agents.Tracks:
    shape = (n, frames)
    return agents.Tracks(
        [f"car-{k:04d}" for k in range(n)], np.arange(frames) * 0.5,
        np.arange(n * frames, dtype=float).reshape(shape), np.zeros(shape), np.zeros(shape),
        np.full(shape, 5.0),
    )  # fmt: skip


def test_detections_are_the_visible_pixels_of_each_car(tmp_path: Path) -> None:
    files = []
    for frame in range(2):
        sem, inst = np.zeros((20, 30), np.uint8), np.zeros((20, 30), int)
        sem[2:5, 3:9], inst[2:5, 3:9] = 4, 11  # first_instance 10: instance 11 is car 0
        if frame == 1:
            sem[10:12, 20:22], inst[10:12, 20:22] = 4, 13  # car 2 appears in frame 1 only
        sem[15:18, 0:4], inst[15:18, 0:4] = 4, 5  # a parked car (instance 5): not tracked
        sem[0:2, 25:28], inst[0:2, 25:28] = 2, 11  # a building with the same number
        files.append(tmp_path / f"f{frame}.png")
        _write(files[-1], sem, inst)
    out = tracks.from_labels(files, _tracks(), first_instance=10)
    by_id = {t["id"]: t for t in out["tracks"]}
    assert [t["instance"] for t in out["tracks"]] == [11, 12, 13]
    car0, car1, car2 = by_id["car-0000"], by_id["car-0001"], by_id["car-0002"]
    assert [d["frame"] for d in car0["detections"]] == [0, 1]
    assert car0["detections"][0]["bbox"] == [3, 2, 8, 4] and car0["detections"][0]["pixels"] == 18
    assert car1["detections"] == []  # hidden or off screen in both frames
    assert [d["frame"] for d in car2["detections"]] == [1]
    assert car2["detections"][0]["bbox"] == [20, 10, 21, 11] and car2["detections"][0]["t"] == 0.5
    assert car2["detections"][0]["x"] == 5.0 and car2["detections"][0]["speed"] == 5.0


def test_the_number_of_label_images_must_match_the_track_frames(tmp_path: Path) -> None:
    _write(tmp_path / "a.png", np.zeros((4, 4), np.uint8), np.zeros((4, 4), int))
    with pytest.raises(ValueError, match="1 label images for 2"):
        tracks.from_labels([tmp_path / "a.png"], _tracks(), 0)
