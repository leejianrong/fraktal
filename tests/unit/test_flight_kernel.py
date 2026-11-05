import math

import numpy as np
import pytest

from fraktal.kernels.flight import fly

L_SHAPE = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0]])


def _run(**kw: float) -> dict[str, np.ndarray]:
    args = {"cruise_mps": 20.0, "max_accel": 3.0, "max_yaw_rate_dps": 60.0, "fps": 10.0}
    return fly(L_SHAPE, **{**args, **kw})  # type: ignore[arg-type]


def test_golden_the_drone_flies_an_l_and_stops_at_the_end() -> None:
    out = _run()
    assert out["xy"][0].tolist() == [0.0, 0.0] and out["speed"][0] == 0.0
    assert out["xy"][-1] == pytest.approx([200.0, 200.0], abs=2.0)
    assert out["speed"][-1] < 0.3
    assert out["speed"].max() == pytest.approx(20.0, abs=2.0)  # reaches cruise speed
    assert 25.0 < out["t"][-1] < 45.0  # about 400 m at up to 20 m/s, plus the start and stop


def test_acceleration_and_yaw_rate_stay_within_their_limits() -> None:
    out = _run(max_accel=2.0, max_yaw_rate_dps=30.0)
    dt = float(out["t"][1] - out["t"][0])
    v = np.diff(out["xy"], axis=0) / dt
    accel = np.hypot(*np.diff(v, axis=0).T) / dt
    assert accel.max() < 2.0 * 1.15  # the cap, with a little room for the finite step
    yaw = np.degrees(np.unwrap(np.radians(out["yaw_deg"])))
    assert np.abs(np.diff(yaw)).max() / dt < 30.0 * 1.05


def test_the_corner_is_rounded_not_cut_sharply_and_not_overshot() -> None:
    out = _run()
    xy = out["xy"]
    near = np.hypot(xy[:, 0] - 200.0, xy[:, 1] - 0.0)
    assert 0.5 < near.min() < 40.0  # it cuts the corner a little
    assert xy[:, 0].max() < 215.0 and xy[:, 1].min() > -15.0  # but does not fly far off the path


def test_the_same_inputs_give_the_same_flight() -> None:
    a, b = _run(), _run()
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize(
    "kw", [{"cruise_mps": 0.0}, {"max_accel": -1.0}, {"max_yaw_rate_dps": 0.0}, {"fps": 0.0}]
)
def test_bad_limits_are_refused(kw: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        _run(**kw)


def test_bad_waypoints_are_refused() -> None:
    args = {"cruise_mps": 5.0, "max_accel": 2.0, "max_yaw_rate_dps": 40.0, "fps": 5.0}
    with pytest.raises(ValueError, match="N >= 2"):
        fly(np.zeros((1, 2)), **args)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same point"):
        fly(np.zeros((3, 2)), **args)  # type: ignore[arg-type]


def test_a_straight_leg_reaches_the_distance_speed_and_accel_say() -> None:
    out = fly(np.array([[0.0, 0.0], [0.0, 400.0]]), cruise_mps=20.0, max_accel=4.0,
              max_yaw_rate_dps=60.0, fps=10.0)  # fmt: skip
    assert out["yaw_deg"][0] == pytest.approx(0.0) and math.isclose(
        out["xy"][-1][1], 400.0, abs_tol=2.0
    )
