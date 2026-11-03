import numpy as np
import pytest

from fraktal.kernels.noise import lattice_shape, value_noise


def test_golden_output_two_by_two_lattice() -> None:
    lattice = np.array([[0.0, 1.0, 9.0], [1.0, 0.0, 9.0], [9.0, 9.0, 9.0]])  # 9s are unused
    got = value_noise(lattice, 2.0, (3, 3))
    # Nodes at rows/cols 0 and 2; the centre pixel (1, 1) is the smoothstep midpoint, 0.5.
    expected = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_reproduces_lattice_values_at_nodes() -> None:
    rng = np.random.default_rng(0)
    lattice = rng.normal(size=(7, 7))
    got = value_noise(lattice, 4.0, (21, 21))
    np.testing.assert_allclose(got[::4, ::4], lattice[:6, :6], atol=1e-5)


def test_stays_within_lattice_range() -> None:
    lattice = np.random.default_rng(1).uniform(-1, 1, size=(10, 10))
    got = value_noise(lattice, 3.3, (30, 30))
    assert got.min() >= lattice.min() - 1e-6
    assert got.max() <= lattice.max() + 1e-6


def test_constant_lattice_gives_constant_noise() -> None:
    assert value_noise(np.full((4, 4), 0.25), 2.5, (7, 7)) == pytest.approx(0.25)


def test_is_a_pure_function_of_its_inputs() -> None:
    lattice = np.random.default_rng(2).normal(size=(6, 6))
    a = value_noise(lattice, 2.0, (9, 9))
    b = value_noise(lattice.copy(), 2.0, (9, 9))
    assert (a == b).all()


def test_rejects_a_lattice_that_is_too_small() -> None:
    assert lattice_shape((9, 9), 2.0) == (6, 6)
    with pytest.raises(ValueError, match="lattice must be at least"):
        value_noise(np.zeros((3, 3)), 2.0, (9, 9))
    with pytest.raises(ValueError):
        value_noise(np.zeros((5, 5)), 0.0, (3, 3))
