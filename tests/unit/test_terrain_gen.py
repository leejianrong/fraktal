import numpy as np
import pytest

from fraktal.terrain_gen import add_detail, fractal_terrain


def blocks(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x.reshape(30, 5, 30, 5).mean(axis=(1, 3))
    return out


def test_fractal_terrain_is_deterministic_seeded_and_has_the_requested_relief() -> None:
    a = fractal_terrain((200, 200), 5.0, 1, relief_m=40.0)
    assert (a == fractal_terrain((200, 200), 5.0, 1, relief_m=40.0)).all()
    assert not np.allclose(a, fractal_terrain((200, 200), 5.0, 2, relief_m=40.0))
    assert a.dtype == np.float32 and 10 < np.ptp(a) <= 40.01
    assert np.ptp(fractal_terrain((200, 200), 5.0, 1, relief_m=80.0)) == pytest.approx(
        2 * np.ptp(a), rel=1e-4
    )


def test_fractal_terrain_has_structure_at_every_scale() -> None:
    a = fractal_terrain(
        (400, 400), 2.0, 3, relief_m=40.0, largest_cell_m=400.0, smallest_cell_m=4.0
    )
    steps = np.abs(np.diff(a, axis=1))
    assert steps.max() < 8 and steps.mean() > 0.01  # smooth but not flat


def test_add_detail_keeps_the_large_scale_shape_and_adds_fine_relief() -> None:
    base = np.tile(np.linspace(300, 320, 150, dtype=np.float32), (150, 1))
    detailed = add_detail(base, 2.0, 5, amplitude_m=2.0)
    assert np.abs(detailed - base).max() <= 1.0 + 1e-4  # half the peak-to-peak amplitude
    assert np.abs(np.diff(detailed, axis=0)).mean() > 0 == np.abs(np.diff(base, axis=0)).mean()
    assert np.abs(blocks(detailed) - blocks(base)).mean() < 0.5
    assert (add_detail(base, 2.0, 5) == add_detail(base, 2.0, 5)).all()


def test_bad_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="persistence"):
        fractal_terrain((50, 50), 1.0, 0, persistence=1.5)
    with pytest.raises(ValueError, match="cell sizes"):
        fractal_terrain((50, 50), 1.0, 0, largest_cell_m=2.0, smallest_cell_m=8.0)
