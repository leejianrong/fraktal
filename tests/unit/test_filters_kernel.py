import numpy as np
import pytest

from fraktal.kernels.filters import gaussian_blur, gaussian_kernel


def test_golden_kernel_for_sigma_one() -> None:
    expected = [0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044]
    got = gaussian_kernel(1.0)
    assert got.size == 7
    np.testing.assert_allclose(got, expected, atol=2e-3)
    assert got.sum() == pytest.approx(1.0, abs=1e-6)


def test_a_point_spreads_into_the_kernel_and_keeps_its_total() -> None:
    img = np.zeros((11, 11), dtype="float32")
    img[5, 5] = 1.0
    out = gaussian_blur(img, 1.0)
    assert out.sum() == pytest.approx(1.0, abs=1e-5)
    k = gaussian_kernel(1.0)
    np.testing.assert_allclose(out[5, 2:9], k[3] * k, atol=1e-6)


def test_sigma_zero_is_identity_and_constants_stay_constant() -> None:
    img = np.random.default_rng(0).uniform(size=(6, 6)).astype("float32")
    assert (gaussian_blur(img, 0.0) == img).all()
    np.testing.assert_allclose(gaussian_blur(np.full((9, 9), 0.4, "float32"), 2.0), 0.4, atol=1e-6)


def test_blurs_each_band_of_a_stack_independently() -> None:
    stack = np.zeros((3, 9, 9), dtype="float32")
    stack[1, 4, 4] = 1.0
    out = gaussian_blur(stack, 1.0)
    assert out[0].sum() == 0 and out[2].sum() == 0
    assert out[1].sum() == pytest.approx(1.0, abs=1e-5)


def test_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError):
        gaussian_kernel(-1.0)
