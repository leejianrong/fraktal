import hashlib

import numpy as np

from fraktal.ids import substream
from fraktal.kernels.sensor import add_gaussian_noise, grade, sensor_chain


def _image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (3, 24, 32), dtype=np.uint8)


def _digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def test_sensor_chain_golden_output() -> None:
    out = sensor_chain(
        _image(), haze=(0.2, 50.0), tone=(0.92, 1.0, 0.86), contrast=1.2, psf_sigma_px=0.8
    )
    assert out.shape == (3, 24, 32) and out.dtype == np.uint8
    assert _digest(out) == "3104cb376462987d"


def test_noise_golden_output_and_determinism() -> None:
    a = add_gaussian_noise(_image(), 3.0, substream(0, "sensor:noise:t"))
    b = add_gaussian_noise(_image(), 3.0, substream(0, "sensor:noise:t"))
    assert np.array_equal(a, b)
    assert _digest(a) == "d6c05bef2a7088b6"


def test_neutral_settings_change_nothing() -> None:
    image = _image()
    assert np.array_equal(sensor_chain(image), image)
    radiance = np.moveaxis(image.astype("float32"), 0, -1)
    assert np.array_equal(grade(radiance), radiance)
