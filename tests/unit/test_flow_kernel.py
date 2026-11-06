import numpy as np
import pytest

from fraktal import depth
from fraktal.kernels import flow


def _cams(dx: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pose = {"x": 0.0, "y": 0.0, "z": 100.0, "heading_deg": 0.0, "off_nadir_deg": 0.0}
    a = depth.camera_to_world(pose)
    b = depth.camera_to_world({**pose, "x": dx})
    return a, b, depth.intrinsics(40.0, 64)


def test_no_camera_motion_means_no_flow() -> None:
    a, _, k = _cams()
    out = flow.flow(np.full((64, 64), 100.0), a, a, k)
    assert np.abs(out).max() == pytest.approx(0.0, abs=1e-4)


def test_a_camera_moving_east_over_flat_ground_shifts_the_image_west_by_f_dx_over_z() -> None:
    a, b, k = _cams(dx=2.0)
    out = flow.flow(np.full((64, 64), 100.0), a, b, k)
    assert out[..., 0] == pytest.approx(-k["fx"] * 2.0 / 100.0, abs=1e-4)
    assert out[..., 1] == pytest.approx(0.0, abs=1e-4)


def test_a_point_that_moves_with_the_camera_has_no_flow_and_sky_is_nan() -> None:
    a, b, k = _cams(dx=2.0)
    z = np.full((64, 64), 100.0)
    z[0, 0] = np.inf
    motion = np.zeros((64, 64, 3))
    motion[..., 0] = 2.0
    out = flow.flow(z, a, b, k, motion)
    assert np.nanmax(np.abs(out)) == pytest.approx(0.0, abs=1e-4)
    assert np.isnan(out[0, 0]).all()
