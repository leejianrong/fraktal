from pathlib import Path

import numpy as np
import pytest
import rasterio

from fraktal import camera, depth, flowgt


def _png(path: Path, data: np.ndarray) -> None:
    with rasterio.open(path, "w", driver="PNG", count=1, dtype="uint16", height=32, width=32) as d:
        d.write(data[None])


def test_a_sequence_gets_one_flow_file_per_pair_and_matches_the_kernel(tmp_path: Path) -> None:
    poses = camera.trajectory("flight", 1000.0, 1, altitude_m=100.0, off_nadir_deg=0.0,
                              duration_s=1.0, fps=4.0, speed_mps=10.0)  # fmt: skip
    k = depth.intrinsics(40.0, 32)
    table = {"intrinsics": k, "frames": [
        {"index": p.index, "camera_to_world": depth.camera_to_world(p.as_dict()).tolist()}
        for p in poses]}  # fmt: skip
    files = []
    for i in range(len(poses)):
        f = tmp_path / f"d{i}.png"
        _png(f, np.full((32, 32), round(100.0 / depth.UNIT_M), dtype="uint16"))
        files.append(f)
    n = flowgt.write_sequence(tmp_path, table, files)
    assert n == len(poses) - 1 == len(list((tmp_path / "flow").glob("*.npy")))
    field = np.load(tmp_path / "flow" / "frame_0000.npy").astype("float32")
    assert field.shape == (32, 32, 2)
    step = 10.0 / 3  # a 10 m run over three frame gaps
    assert np.hypot(*field.mean(axis=(0, 1))) == pytest.approx(k["fx"] * step / 100.0, rel=0.02)


def test_a_depth_image_is_needed_for_every_pose(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        flowgt.write_sequence(tmp_path, {"intrinsics": {}, "frames": [{}, {}]}, [])
