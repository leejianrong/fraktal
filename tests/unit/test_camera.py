import math

import pytest

from fraktal import camera


def test_the_same_seed_gives_the_same_poses_and_another_seed_does_not() -> None:
    a = camera.trajectory("flight", 500.0, 3)
    b = camera.trajectory("flight", 500.0, 3)
    c = camera.trajectory("flight", 500.0, 4)
    assert a == b and a != c


def test_frame_count_and_timing_follow_duration_and_fps() -> None:
    poses = camera.trajectory("flight", 500.0, 0, duration_s=2.0, fps=12.0)
    assert len(poses) == 24 and poses[0].t_s == 0.0
    assert poses[1].t_s == pytest.approx(1 / 12)
    assert len(camera.trajectory("still", 500.0, 0)) == 1


def test_a_pose_looks_at_its_target_at_the_requested_angle() -> None:
    for kind in camera.KINDS:
        for p in camera.trajectory(kind, 500.0, 5, altitude_m=300.0, off_nadir_deg=35.0):
            ground = math.hypot(p.target_x - p.x, p.target_y - p.y)
            assert math.degrees(math.atan2(ground, p.z)) == pytest.approx(35.0, abs=0.01)
            heading = math.degrees(math.atan2(p.target_x - p.x, p.target_y - p.y)) % 360.0
            assert heading == pytest.approx(p.heading_deg, abs=0.01)


def test_a_nadir_view_sits_right_above_its_target() -> None:
    (p,) = camera.trajectory("still", 400.0, 1, off_nadir_deg=0.0)
    assert (p.x, p.y) == (p.target_x, p.target_y) == (200.0, 200.0)


def test_an_orbit_turns_through_the_sweep_around_the_centre() -> None:
    poses = camera.trajectory("orbit", 500.0, 2, sweep_deg=90.0)
    assert {(p.target_x, p.target_y) for p in poses} == {(250.0, 250.0)}
    assert (poses[-1].heading_deg - poses[0].heading_deg) % 360.0 == pytest.approx(90.0)


def test_a_flight_is_a_line_over_the_centre_at_the_set_speed() -> None:
    poses = camera.trajectory("flight", 500.0, 7, speed_mps=40.0, duration_s=2.0)
    run = math.hypot(poses[-1].target_x - poses[0].target_x, poses[-1].target_y - poses[0].target_y)
    assert run == pytest.approx(80.0)
    mid = (
        (poses[0].target_x + poses[-1].target_x) / 2,
        (poses[0].target_y + poses[-1].target_y) / 2,
    )
    assert mid == pytest.approx((250.0, 250.0))


def test_a_flight_that_leaves_the_tile_is_refused() -> None:
    with pytest.raises(ValueError, match="outside the tile"):
        camera.trajectory("flight", 100.0, 0, speed_mps=100.0)


@pytest.mark.parametrize("bad", [{"off_nadir_deg": 85.0}, {"off_nadir_deg": -1.0}])
def test_a_bad_angle_is_refused(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="off-nadir"):
        camera.trajectory("still", 500.0, 0, **bad)


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown trajectory"):
        camera.trajectory("spiral", 500.0, 0)
