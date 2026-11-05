import numpy as np
import pytest

from fraktal.kernels.traffic import braking_profile, simulate


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
