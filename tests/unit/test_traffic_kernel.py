import numpy as np
import pytest

from fraktal.kernels.traffic import braking_profile, simulate, simulate_interacting


def test_braking_profile_slows_the_approach_to_a_slow_point() -> None:
    cap = np.full((1, 200), 15.0)
    cap[0, 100] = 2.0
    out = braking_profile(cap, 1.0, 2.0)
    assert out[0, 100] == 2.0 and out[0, 0] == 15.0
    assert out[0, 90] < out[0, 80] < out[0, 60]  # lower the nearer the slow point
    assert out[0, 99] == pytest.approx(np.sqrt(2.0**2 + 2 * 2.0 * 1.0))  # one metre of braking


def test_golden_a_car_accelerates_to_the_cap_and_holds_it() -> None:
    profile = np.full((1, 600), 10.0)
    s, v = simulate(profile, np.array([600.0]), 1.0, np.array([0.0]), np.array([0.0]),
                    accel=2.0, brake=3.0, dt=0.1, steps=300)  # fmt: skip
    assert v[0, 0] == 0.0 and v[0, 50] == pytest.approx(10.0, abs=0.1)  # 2 m/s^2 for 5 s
    assert v[0, -1] == pytest.approx(10.0, abs=1e-9)
    assert s[0, -1] == pytest.approx(s[0, 50] + 10.0 * 25.0, rel=0.01)


def test_a_car_brakes_for_a_slow_section_and_stops_at_the_end_of_its_route() -> None:
    cap = np.full((1, 300), 12.0)
    cap[0, 150:160] = 3.0
    profile = braking_profile(cap, 1.0, 2.8)
    s, v = simulate(profile, np.array([300.0]), 1.0, np.array([0.0]), np.array([12.0]),
                    accel=2.5, brake=3.5, dt=0.1, steps=600)  # fmt: skip
    at_slow = v[0, np.argmin(np.abs(s[0] - 150.0))]
    assert at_slow < 4.0  # slowed in time
    assert (np.diff(s[0]) >= 0).all() and s[0, -1] <= 300.0  # never backwards, never past the end


def test_bad_input_is_refused() -> None:
    with pytest.raises(ValueError, match="2D"):
        braking_profile(np.ones(5), 1.0, 1.0)
    with pytest.raises(ValueError, match="positive"):
        braking_profile(np.ones((1, 5)), 0.0, 1.0)
    with pytest.raises(ValueError, match="one row"):
        simulate(np.ones((2, 5)), np.ones(1), 1.0, np.zeros(2), np.zeros(2),
                 accel=1.0, brake=1.0, dt=0.1, steps=3)  # fmt: skip


def _straight(n: int, length: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` copies of an eastbound road, one grid point a metre: xy, heading, rank."""
    xy = np.zeros((n, length, 2))
    xy[:, :, 0] = np.arange(length)
    return xy, np.zeros((n, length)), np.zeros((n, length))


PARAMS = {"accel": 2.5, "comfort_brake": 2.5, "max_brake": 8.0, "dt": 0.1}


def test_a_fast_car_queues_behind_a_slow_one_and_never_touches_it() -> None:
    xy, head, rank = _straight(2)
    profile = np.vstack([np.full(400, 4.0), np.full(400, 12.0)])  # car 0 is slow, car 1 wants 12
    s, v = simulate_interacting(
        profile,
        xy,
        head,
        rank,
        np.full(2, 400.0),
        1.0,
        np.array([40.0, 0.0]),
        np.array([4.0, 12.0]),
        steps=400,
        **PARAMS,
    )
    gap = s[0] - s[1]
    assert gap.min() > 4.5 + 0.5  # never closer than a car length plus a little
    assert v[1, -1] == pytest.approx(4.0, abs=0.6)  # it has settled at the leader's speed
    assert v[1, :150].max() > 6.0  # but it drove faster before catching up
    free, _ = simulate(
        profile,
        np.full(2, 400.0),
        1.0,
        np.array([40.0, 0.0]),
        np.array([4.0, 12.0]),
        accel=2.5,
        brake=8.0,
        dt=0.1,
        steps=400,
    )
    assert (free[0] - free[1]).min() < 0  # without interaction it drives straight through


def test_a_car_stops_for_a_stopped_leader_and_a_free_car_keeps_its_speed() -> None:
    xy, head, rank = _straight(3, 600)
    profile = np.vstack([np.zeros(600), np.full(600, 10.0), np.full(600, 10.0)])
    xy[2, :, 1] = 40.0  # car 2 is on another road, far away
    s, v = simulate_interacting(
        profile,
        xy,
        head,
        rank,
        np.full(3, 600.0),
        1.0,
        np.array([200.0, 0.0, 0.0]),
        np.array([0.0, 10.0, 10.0]),
        steps=500,
        **PARAMS,
    )
    assert s[0, -1] - s[1, -1] > 4.5 and v[1, -1] < 0.5  # stopped behind it, short of a collision
    assert v[2, 300] == pytest.approx(10.0, abs=0.3)  # the other car was not held up


def test_cars_going_opposite_ways_pass_without_braking() -> None:
    xy, head, rank = _straight(2)
    xy[1] = xy[0, ::-1] + np.array([0.0, 3.0])  # eastbound and westbound lanes, 3 m apart
    head[1] = np.pi
    profile = np.full((2, 400), 10.0)
    s, v = simulate_interacting(
        profile,
        xy,
        head,
        rank,
        np.full(2, 400.0),
        1.0,
        np.array([50.0, 50.0]),
        np.array([10.0, 10.0]),
        steps=200,
        **PARAMS,
    )
    assert v.min() > 9.0


def _crossing(rank_east: float, rank_south: float) -> tuple[np.ndarray, ...]:
    xy = np.zeros((2, 300, 2))
    xy[0, :, 0], xy[0, :, 1] = np.arange(300) - 100.0, 0.0  # eastbound through the origin
    xy[1, :, 0], xy[1, :, 1] = 0.0, 100.0 - np.arange(300)  # southbound through the origin
    head = np.zeros((2, 300))
    head[1] = -np.pi / 2
    rank = np.vstack([np.full(300, rank_east), np.full(300, rank_south)])
    return xy, head, rank


def test_the_lower_priority_car_gives_way_at_a_crossing() -> None:
    for east, south, loser in ((1.0, 0.0, 1), (0.0, 1.0, 0), (0.0, 0.0, 1)):
        xy, head, rank = _crossing(east, south)
        profile = np.full((2, 300), 10.0)
        s, v = simulate_interacting(
            profile,
            xy,
            head,
            rank,
            np.full(2, 300.0),
            1.0,
            np.array([60.0, 60.0]),
            np.array([10.0, 10.0]),
            steps=300,
            **PARAMS,
        )
        pos = np.stack(
            [xy[0, np.minimum(s[0].astype(int), 299)], xy[1, np.minimum(s[1].astype(int), 299)]]
        )
        apart = np.hypot(*(pos[0] - pos[1]).T)
        assert apart.min() > 3.0  # they never meet in the middle
        assert v[loser].min() < 8.0 and v[1 - loser].min() > 9.0  # only the loser slowed


def test_a_car_that_has_waited_long_enough_goes_on_so_a_junction_never_locks() -> None:
    xy, head, rank = _crossing(0.0, 0.0)
    profile = np.full((2, 300), 10.0)
    s, _ = simulate_interacting(
        profile,
        xy,
        head,
        rank,
        np.full(2, 300.0),
        1.0,
        np.array([40.0, 40.0]),
        np.array([0.0, 0.0]),
        steps=900,
        **PARAMS,
    )
    assert s[:, -1].min() > 100.0  # both got through


def test_the_interacting_simulation_checks_its_input() -> None:
    xy, head, rank = _straight(2)
    with pytest.raises(ValueError, match="one row"):
        simulate_interacting(
            np.ones((3, 400)),
            xy,
            head,
            rank,
            np.ones(2),
            1.0,
            np.zeros(2),
            np.zeros(2),
            steps=3,
            **PARAMS,
        )
    with pytest.raises(ValueError, match="positive"):
        simulate_interacting(
            np.ones((2, 400)),
            xy,
            head,
            rank,
            np.ones(2),
            1.0,
            np.zeros(2),
            np.zeros(2),
            steps=3,
            **{**PARAMS, "dt": 0.0},
        )
