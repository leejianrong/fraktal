import numpy as np
import pytest

from fraktal.kernels.hydrology import DC, DR, flow_accumulation


def test_golden_tilted_plane_drains_south() -> None:
    dem = np.array([[10.0] * 4, [9.0] * 4, [8.0] * 4])  # falls toward the bottom row
    acc, direction = flow_accumulation(dem)
    assert acc.tolist() == [[1.0] * 4, [2.0] * 4, [3.0] * 4]
    assert (direction[:2] == 4).all() and (direction[2] == -1).all()  # 4 is south, -1 an outlet


def test_a_valley_collects_both_slopes() -> None:
    cols = np.abs(np.arange(7) - 3.0)  # V shape: lowest in column 3
    dem = np.tile(cols, (5, 1)) * 2 + np.arange(5)[::-1, None] * 0.1  # gentle fall to the south
    acc, _ = flow_accumulation(dem)
    assert acc[4].argmax() == 3
    assert acc[4, 3] > 3 * acc[4, 0]


def test_all_water_ends_at_an_outlet() -> None:
    rng = np.random.default_rng(0)
    dem = rng.uniform(0, 100, (30, 30))
    acc, direction = flow_accumulation(dem)
    assert acc[direction == -1].sum() == pytest.approx(dem.size)  # mass is conserved
    assert acc.min() >= 1


def test_direction_points_to_a_lower_neighbour() -> None:
    rng = np.random.default_rng(1)
    dem = rng.uniform(0, 10, (12, 12))
    _, direction = flow_accumulation(dem)
    for r in range(12):
        for c in range(12):
            k = direction[r, c]
            if k >= 0:
                assert dem[r + DR[k], c + DC[k]] < dem[r, c]


def test_is_deterministic_on_flats_and_rejects_bad_input() -> None:
    flat = np.zeros((6, 6))
    a, da = flow_accumulation(flat)
    b, db = flow_accumulation(flat)
    assert (a == b).all() and (da == db).all() and (da == -1).all()
    with pytest.raises(ValueError):
        flow_accumulation(np.zeros(5))
