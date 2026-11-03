import numpy as np
import pytest

from fraktal.kernels.pathing import least_cost_path


def test_golden_path_goes_through_the_gap_in_a_wall() -> None:
    cost = np.ones((5, 7))
    cost[:, 3] = np.inf
    cost[2, 3] = 1.0  # the only gap
    path = least_cost_path(cost, (2, 0), (2, 6))
    assert path is not None
    assert path.tolist() == [[2, c] for c in range(7)]


def test_prefers_a_longer_cheap_route_over_a_short_expensive_one() -> None:
    cost = np.ones((5, 9))
    cost[2, 2:7] = 50.0  # a costly strip straight ahead
    path = least_cost_path(cost, (2, 0), (2, 8))
    assert path is not None and not any(r == 2 and 2 <= c <= 6 for r, c in path.tolist())


def test_unreachable_goal_and_bad_input() -> None:
    cost = np.ones((4, 4))
    cost[:, 2] = np.inf
    assert least_cost_path(cost, (0, 0), (0, 3)) is None
    with pytest.raises(ValueError, match="outside"):
        least_cost_path(cost, (0, 0), (9, 9))
    with pytest.raises(ValueError, match="2D"):
        least_cost_path(np.ones(4), (0, 0), (0, 1))
    assert least_cost_path(cost, (1, 1), (1, 1)).tolist() == [[1, 1]]  # type: ignore[union-attr]


def test_is_deterministic_on_ties_and_each_step_is_adjacent() -> None:
    cost = np.ones((12, 12))
    a = least_cost_path(cost, (0, 0), (11, 11))
    b = least_cost_path(cost, (0, 0), (11, 11))
    assert a is not None and b is not None and (a == b).all()
    assert (np.abs(np.diff(a, axis=0)).max(axis=1) == 1).all()
    assert len(a) == 12  # the diagonal
