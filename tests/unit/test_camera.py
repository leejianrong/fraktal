import math
from typing import Any

import numpy as np
import pytest

from fraktal import camera

ROUTE = [(150.0, 150.0), (200.0, 160.0), (230.0, 200.0)]


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
        for p in camera.trajectory(
            kind, 500.0, 5, altitude_m=300.0, off_nadir_deg=35.0, waypoints=ROUTE
        ):
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
def test_a_bad_angle_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="off-nadir"):
        camera.trajectory("still", 500.0, 0, **bad)


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown trajectory"):
        camera.trajectory("spiral", 500.0, 0)


def test_a_path_starts_and_ends_on_the_waypoints_and_turns_gradually() -> None:
    poses = camera.trajectory("path", 500.0, 0, waypoints=ROUTE, speed_mps=20.0, fps=10.0,
                              off_nadir_deg=0.0)  # fmt: skip
    assert (poses[0].x, poses[0].y) == pytest.approx((150.0, 150.0), abs=1e-3)
    assert (poses[-1].x, poses[-1].y) == pytest.approx((230.0, 200.0), abs=2.0)
    assert poses[0].heading_deg == pytest.approx(math.degrees(math.atan2(50, 10)), abs=2.0)
    turns = [abs(b.heading_deg - a.heading_deg) for a, b in zip(poses, poses[1:], strict=False)]
    assert max(t for t in turns if t < 180) < 60.0 / 10.0 + 0.01  # at most the yaw rate per frame
    assert camera.trajectory("path", 500.0, 0, waypoints=ROUTE) == camera.trajectory(
        "path", 500.0, 0, waypoints=ROUTE
    )


def test_a_path_is_checked() -> None:
    with pytest.raises(ValueError, match="at least two"):
        camera.trajectory("path", 500.0, 0, waypoints=[(1.0, 1.0)])
    with pytest.raises(ValueError, match="at least two"):
        camera.trajectory("path", 500.0, 0)
    with pytest.raises(ValueError, match="outside the tile"):
        camera.trajectory("path", 500.0, 0, waypoints=[(10.0, 10.0), (490.0, 490.0)])


def test_a_riding_camera_sits_on_the_agent_looking_along_its_heading() -> None:
    import numpy as np

    from fraktal import depth

    t = np.arange(3) * 0.5
    mount = {"mount_height_m": 1.25, "forward_m": 0.9, "pitch_deg": 3.0, "fov_deg": 90.0}
    east = camera.follow_poses([10.0, 12.0, 14.0], [5.0] * 3, [0.0] * 3, t, **mount)
    assert [p.heading_deg for p in east] == [90.0] * 3  # east is 90 degrees clockwise from north
    assert east[0].x == pytest.approx(10.9) and east[0].y == pytest.approx(5.0)
    assert east[0].z == pytest.approx(1.25) and east[0].off_nadir_deg == pytest.approx(87.0)
    m = depth.camera_to_world(east[0].as_dict())
    forward = m[:3, 2]  # camera z axis: where it looks
    assert forward[0] > 0.99 and forward[2] < 0 and abs(forward[1]) < 1e-9  # east, a little down
    north = camera.follow_poses([0.0], [0.0], [np.pi / 2], [0.0], mount_height_m=1.6,
                                forward_m=0.0, pitch_deg=0.0, fov_deg=70.0)  # fmt: skip
    assert north[0].heading_deg == pytest.approx(0.0)
    assert depth.camera_to_world(north[0].as_dict())[:3, 2] == pytest.approx([0.0, 1.0, 0.0])


def test_a_riding_camera_follows_the_ground_and_checks_its_pitch() -> None:
    import numpy as np

    (p,) = camera.follow_poses([4.0], [6.0], [0.0], [0.0], mount_height_m=1.5, forward_m=0.0,
                               pitch_deg=0.0, fov_deg=90.0, ground=lambda x, y: x + y)  # fmt: skip
    assert p.z == pytest.approx(1.5 + 10.0)
    with pytest.raises(ValueError, match="pitch"):
        camera.follow_poses(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), mount_height_m=1.0,
                            forward_m=0.0, pitch_deg=75.0, fov_deg=90.0)  # fmt: skip


def HILL(x: Any, y: Any) -> Any:  # noqa: N802  a 1:2 slope rising east
    return 0.5 * np.asarray(x, dtype="float64")


def _flat(x: Any, y: Any) -> Any:
    return np.zeros_like(np.asarray(x, dtype="float64"))


def test_a_path_with_ground_flies_above_the_terrain_it_crosses() -> None:
    route = [(100.0, 250.0), (400.0, 250.0)]
    flat = camera.trajectory("path", 500.0, 0, waypoints=route, altitude_m=40.0, speed_mps=20.0)
    assert {p.z for p in flat} == {40.0}  # the old behaviour: absolute altitude
    over = camera.trajectory(
        "path", 500.0, 0, waypoints=route, altitude_m=40.0, speed_mps=20.0, ground=HILL
    )
    assert all(p.z >= HILL(p.x, p.y) + 40.0 for p in over)
    assert any(p.z < HILL(p.x, p.y) for p in flat)  # the failure this fixes: under the hillside


def test_an_eye_level_walk_keeps_eye_height_over_a_slope() -> None:
    poses, stats = camera.street_poses(
        [(10.0, 20.0), (60.0, 20.0)],
        speed_mps=1.4,
        fps=4.0,
        ground=HILL,
        hazards=camera.obstacles([], []),
    )
    assert len(poses) > 30 and stats == {"blocked_share": 0.0, "pushed_m": 0.0}
    assert all(p.z >= HILL(p.x, p.y) + 1.6 for p in poses)  # the highest ground nearby, so more
    assert all(p.heading_deg == pytest.approx(90.0, abs=1.0) for p in poses)  # along the route
    assert all(p.off_nadir_deg == pytest.approx(90.0) for p in poses)  # level


def test_an_eye_level_walk_stays_out_of_buildings_and_walls() -> None:
    house = {"ring": [[28.0, 18.0], [34.0, 18.0], [34.0, 24.0], [28.0, 24.0], [28.0, 18.0]]}
    wall = [20.0, 14.0, 44.0, 14.0]
    hazards = camera.obstacles([house], [wall])
    route = [(10.0, 20.0), (60.0, 20.0)]  # straight through the house
    walked, stats = camera.street_poses(
        route, speed_mps=1.4, fps=4.0, ground=_flat, hazards=hazards
    )
    assert stats["pushed_m"] > 1.0
    from shapely.geometry import Point

    assert all(hazards.distance(Point(p.x, p.y)) >= 0.99 for p in walked)
    raw, _ = camera.street_poses(
        route, speed_mps=1.4, fps=4.0, ground=_flat, hazards=camera.obstacles([], [])
    )
    assert any(hazards.contains(Point(p.x, p.y)) for p in raw)  # the failure this fixes


def test_a_camera_looking_into_a_wall_is_counted_as_blocked() -> None:
    hazards = camera.obstacles([], [[10.0, -5.0, 10.0, 5.0]])
    share = camera.blocked_share(
        [8.0, 0.0], [0.0, 0.0], [90.0, 90.0], hazards
    )  # east; second is past it
    assert share == 0.5
    assert camera.blocked_share([8.0], [0.0], [270.0], hazards) == 0.0  # facing away
