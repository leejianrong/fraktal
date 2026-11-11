import numpy as np
import pytest

from fraktal.kernels.paddy import FLOODED, HARVESTED, PLANTED, paddy_tone


def _case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ground = np.tile(np.array([0.44, 0.62, 0.24], dtype="float32"), (8, 8, 1))
    cells = np.zeros((8, 8), dtype="int32")
    cells[:, :4], cells[:, 4:6] = 1, 2
    state = np.array([0, PLANTED, FLOODED], dtype="int64")
    jitter = np.array([[0, 0, 0], [0, 0, 0], [0.5, -0.5, 0.25]], dtype="float32")
    return ground, cells, state, jitter, np.zeros(3, dtype="float32")


def test_golden_planted_block_with_no_jitter() -> None:
    ground, cells, state, jitter, angle = _case()
    out = paddy_tone(ground, cells, state, jitter, angle, 0.3)
    np.testing.assert_allclose(out[0, 0], [0.3299, 0.4474, 0.2218], atol=2e-3)  # row phase 0: dark
    assert out[0, 0, 1] < ground[0, 0, 1]  # duller than the class colour


def test_pixels_off_the_blocks_are_unchanged_and_blocks_differ() -> None:
    ground, cells, state, jitter, angle = _case()
    out = paddy_tone(ground, cells, state, jitter, angle, 0.3)
    assert (out[:, 6:] == ground[:, 6:]).all()
    assert not np.allclose(out[0, 0], out[0, 4], atol=0.02)


def test_rows_fade_on_coarse_pixels_and_no_blocks_is_identity() -> None:
    ground, cells, state, jitter, angle = _case()
    fine = paddy_tone(ground, cells, state, jitter, angle, 0.3)[:, :4, 1]
    coarse = paddy_tone(ground, cells, state, jitter, angle, 1.0)[:, :4, 1]
    assert np.ptp(fine) > 0.02 and np.ptp(coarse) < 1e-6
    none = paddy_tone(ground, np.zeros_like(cells), state, jitter, angle, 0.3)
    assert (none == ground).all()


def test_shape_mismatch_is_an_error() -> None:
    ground, cells, state, jitter, angle = _case()
    with pytest.raises(ValueError):
        paddy_tone(ground, cells[:4], state, jitter, angle, 0.3)
    assert HARVESTED == 2
