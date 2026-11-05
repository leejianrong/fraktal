import math

import numpy as np
import pytest

from fraktal import camera, depth


def _pose(**kw: float) -> dict[str, float]:
    base = {"x": 100.0, "y": 50.0, "z": 150.0, "heading_deg": 0.0, "off_nadir_deg": 0.0}
    return {**base, **kw}


def test_intrinsics_put_the_centre_in_the_middle_and_follow_the_field_of_view() -> None:
    k = depth.intrinsics(90.0, 512)
    assert (k["cx"], k["cy"]) == (256.0, 256.0) and k["fx"] == pytest.approx(256.0)
    assert depth.intrinsics(60.0, 512)["fx"] > k["fx"]  # a narrower view magnifies


def test_a_nadir_camera_looks_down_with_north_at_the_top_of_the_image() -> None:
    m = depth.camera_to_world(_pose())
    assert m[:3, 2] == pytest.approx([0, 0, -1])  # forward is down
    assert m[:3, 1] == pytest.approx([0, -1, 0])  # image down is south
    assert m[:3, 0] == pytest.approx([1, 0, 0])  # image right is east


@pytest.mark.parametrize("heading", [0.0, 37.0, 200.0])
@pytest.mark.parametrize("tilt", [0.0, 25.0, 50.0])
def test_a_posed_camera_sees_its_target_at_the_image_centre_at_the_right_depth(
    heading: float, tilt: float
) -> None:
    (p,) = camera.trajectory("still", 1000.0, 3, altitude_m=150.0, off_nadir_deg=tilt)
    pose = {**p.as_dict(), "heading_deg": heading}
    back = 150.0 * math.tan(math.radians(tilt))
    pose["x"] = p.target_x - back * math.sin(math.radians(heading))
    pose["y"] = p.target_y - back * math.cos(math.radians(heading))
    k = depth.intrinsics(40.0, 512)
    u, v, z = depth.project((p.target_x, p.target_y, 0.0), depth.camera_to_world(pose), k)
    assert (u, v) == pytest.approx((256.0, 256.0), abs=1e-6)
    assert z == pytest.approx(150.0 / math.cos(math.radians(tilt)))


def test_a_point_north_of_a_tilted_camera_lands_in_the_upper_half_of_the_image() -> None:
    k = depth.intrinsics(40.0, 512)
    m = depth.camera_to_world(_pose(off_nadir_deg=40.0, y=0.0, x=0.0))
    _, v_far, _ = depth.project((0.0, 400.0, 0.0), m, k)
    _, v_near, _ = depth.project((0.0, 150.0, 0.0), m, k)
    assert v_far < v_near < 512  # farther ground is higher up


def test_a_pixel_ray_and_a_projection_agree() -> None:
    k = depth.intrinsics(50.0, 256)
    m = depth.camera_to_world(_pose(heading_deg=70.0, off_nadir_deg=30.0))
    ray = depth.pixel_to_ray(100.0, 40.0, k)
    point = (m @ np.append(ray * 90.0, 1.0))[:3]
    u, v, z = depth.project(point, m, k)
    assert (u, v, z) == pytest.approx((100.0, 40.0, 90.0))


def test_a_point_behind_the_camera_is_refused() -> None:
    with pytest.raises(depth.DepthError, match="behind"):
        depth.project(
            (0.0, 0.0, 500.0),
            depth.camera_to_world(_pose(x=0.0, y=0.0)),
            depth.intrinsics(40.0, 64),
        )


def test_two_channel_depth_round_trips_to_metres() -> None:
    raw = np.array([[0, 1, 300, 7], [3000, 65535, 20, 8]], dtype=np.uint16)
    rgb = np.stack([raw >> 8, raw & 255, np.zeros_like(raw)], axis=-1).astype(np.uint8)
    assert (depth.from_encoded(rgb) == raw).all()
    assert (depth.from_encoded(np.moveaxis(rgb, -1, 0)) == raw).all()
    metres = depth.to_metres(raw)
    assert np.isinf(metres[0, 0]) and metres[1, 0] == pytest.approx(150.0)
    with pytest.raises(depth.DepthError):
        depth.from_encoded(np.zeros((4, 4)))
