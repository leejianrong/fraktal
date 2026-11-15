"""FRK-171: auto-exposure pulls a strip toward a target with one gamma and leaves labels alone."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from test_render3d import CENTER, fake, world  # noqa: F401

from fraktal import lighting, metering, render3d
from fraktal.kernels import metering as k


def frame(value: int) -> np.ndarray:
    out: np.ndarray = np.full((3, 8, 8), value, dtype="uint8")
    return out


def test_golden_gamma_for_a_flat_dark_frame() -> None:
    hist = k.brightness_histogram(frame(60))
    gamma = k.solve_gamma(hist, 0.46, 0.7, 0.55, 1.3)
    assert gamma == pytest.approx(0.6460, abs=2e-3)  # 0.235 -> 0.393
    assert k.apply_gamma(frame(60), gamma)[0, 0, 0] == 100


def test_strength_zero_changes_nothing_and_one_meets_the_target() -> None:
    hist = k.brightness_histogram(frame(90))
    assert k.solve_gamma(hist, 0.46, 0.0, 0.3, 2.0) == pytest.approx(1.0, abs=1e-6)
    g = k.solve_gamma(hist, 0.46, 1.0, 0.3, 2.0)
    assert k.mean_after(hist, g) == pytest.approx(0.46, abs=1e-3)


def test_the_gamma_stays_inside_its_limits() -> None:
    dark = k.brightness_histogram(frame(5))
    assert k.solve_gamma(dark, 0.46, 1.0, 0.55, 1.3) == pytest.approx(0.55, abs=1e-6)
    white = k.brightness_histogram(frame(250))
    assert k.solve_gamma(white, 0.2, 1.0, 0.55, 1.3) == pytest.approx(1.3, abs=1e-6)


def test_a_dark_and_a_bright_seed_end_closer_together() -> None:
    gaps = []
    for value in (70, 130):  # 0.27 and 0.51 before
        hist = k.brightness_histogram(frame(value))
        g = k.solve_gamma(
            hist, lighting.METER_TARGET, lighting.METER_STRENGTH, *lighting.METER_GAMMA
        )
        gaps.append(k.mean_after(hist, g))
    assert gaps[1] - gaps[0] < 0.51 - 0.27


def test_white_and_black_stay_put() -> None:
    img = np.array([[[0, 255]]], dtype="uint8").repeat(3, axis=0)
    assert k.apply_gamma(img, 0.7).tolist() == img.tolist()


def write(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="PNG", width=data.shape[2], height=data.shape[1], count=3, dtype="uint8"
    ) as dst:
        dst.write(data)


def test_apply_brightens_frames_records_the_note_and_leaves_labels(tmp_path: Path) -> None:
    for i in range(3):
        write(tmp_path / "frames" / f"frame_{i:04d}.png", frame(60))
        write(tmp_path / "labels" / f"frame_{i:04d}.png", frame(1))
    note = metering.apply(tmp_path)
    with rasterio.open(tmp_path / "frames" / "frame_0000.png") as src:
        assert int(src.read()[0, 0, 0]) == 100
    with rasterio.open(tmp_path / "labels" / "frame_0000.png") as src:
        assert int(src.read()[0, 0, 0]) == 1
    assert note["brightness_before"] == pytest.approx(0.2353, abs=1e-3)
    assert json.loads((tmp_path / "metering.json").read_text()) == note


def test_a_strip_without_frames_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        metering.apply(tmp_path)


def test_only_eye_level_strips_are_metered_and_it_can_be_turned_off(
    world: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    opts = {"duration_s": 1.0, "fps": 2.0}
    way = [(20.0, 20.0), (120.0, 40.0)]

    def run(name: str, **kw: object) -> dict[str, object]:
        return render3d.render(
            world, tmp_path / name, CENTER, 300.0, 1.0, seed=4, script=fake(tmp_path),
            camera_options=opts, frame_px=64, **kw,  # type: ignore[arg-type]
        )  # fmt: skip

    street = run("street", trajectory="street", waypoints=way)
    assert street["metering"]["gamma"] < 1.0  # type: ignore[index]
    assert (tmp_path / "street" / "metering.json").is_file()
    assert "metering" not in run("off", trajectory="street", waypoints=way, meter=False)
    assert "metering" not in run("flight", trajectory="flight")
