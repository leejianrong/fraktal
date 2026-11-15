"""FRK-165: the eye-level camera chain is deterministic, off by default and keeps labels exact."""

import json
from pathlib import Path

import numpy as np
import rasterio

from fraktal import eyesensor


def _write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="PNG", width=data.shape[2], height=data.shape[1], count=3, dtype="uint8"
    ) as dst:
        dst.write(data)


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.asarray(src.read())


def _strip(root: Path, n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(3)
    labels = np.zeros((3, 64, 64), dtype="uint8")
    labels[0, :, :] = 1
    labels[0, 20:40, 10:30] = 2  # a building with instance 5
    labels[1, 20:40, 10:30] = 5
    for i in range(n):
        _write(
            root / "frames" / f"frame_{i:04d}.png",
            rng.integers(30, 220, (3, 64, 64), dtype="uint8"),
        )
        _write(root / "labels" / f"frame_{i:04d}.png", labels)
    return labels


def test_the_phone_chain_changes_frames_but_never_the_labels(tmp_path: Path) -> None:
    labels = _strip(tmp_path)
    before = _read(tmp_path / "frames" / "frame_0000.png")
    rig = eyesensor.apply(tmp_path, seed=7)
    after = _read(tmp_path / "frames" / "frame_0000.png")
    assert not np.array_equal(before, after)
    assert np.array_equal(_read(tmp_path / eyesensor.RAW_DIR / "frame_0000.png"), before)
    for i in range(3):
        assert np.array_equal(_read(tmp_path / "labels" / f"frame_{i:04d}.png"), labels)
    assert rig["dashcam"] is None
    assert json.loads((tmp_path / "eye_sensor.json").read_text()) == rig


def test_the_chain_is_deterministic_per_seed(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        _strip(tmp_path / name)
    eyesensor.apply(tmp_path / "a", seed=7)
    eyesensor.apply(tmp_path / "b", seed=7)
    eyesensor.apply(tmp_path / "c", seed=8)
    frame = Path("frames") / "frame_0001.png"
    assert np.array_equal(_read(tmp_path / "a" / frame), _read(tmp_path / "b" / frame))
    assert not np.array_equal(_read(tmp_path / "a" / frame), _read(tmp_path / "c" / frame))


def test_noise_differs_between_frames_of_one_strip(tmp_path: Path) -> None:
    flat = np.full((3, 64, 64), 100, dtype="uint8")
    for i in range(2):
        _write(tmp_path / "frames" / f"frame_{i:04d}.png", flat)
    eyesensor.apply(tmp_path, seed=1)
    a, b = (_read(tmp_path / "frames" / f"frame_{i:04d}.png") for i in range(2))
    assert not np.array_equal(a, b)


def test_a_dashcam_bonnet_blanks_only_the_labels_it_hides(tmp_path: Path) -> None:
    labels = _strip(tmp_path)
    eyesensor.apply(tmp_path, seed=7, dashcam=True)
    got = _read(tmp_path / "labels" / "frame_0000.png")
    hidden = (got == 0).all(axis=0)
    assert hidden[-1].all() and not hidden[:40].any()  # bottom row hidden, the upper scene intact
    assert np.array_equal(got[:, ~hidden], labels[:, ~hidden])  # every other pixel is exact


def test_applying_twice_is_refused(tmp_path: Path) -> None:
    _strip(tmp_path, 1)
    eyesensor.apply(tmp_path, seed=7)
    try:
        eyesensor.apply(tmp_path, seed=7)
    except ValueError as err:
        assert "already applied" in str(err)
    else:
        raise AssertionError("expected a refusal")
