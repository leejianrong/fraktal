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


def _picture() -> np.ndarray:
    rng = np.random.default_rng(5)
    base = np.full((3, 64, 96), 140, dtype="uint8")
    base[:, 20:44, 30:66] = rng.integers(40, 230, (3, 24, 36), dtype="uint8")
    return base


def test_each_lens_effect_is_off_at_zero_and_does_what_it_says() -> None:
    from fraktal.kernels.sensor import lens_effects

    img = _picture()
    assert np.array_equal(lens_effects(img), img)
    dark = lens_effects(img, vignette=0.4)
    assert dark[:, 32, 48].tolist() == img[:, 32, 48].tolist()  # the centre is untouched
    assert dark[:, 0, 0].mean() < 0.65 * img[:, 0, 0].mean() and (dark <= img).all()
    fringed = lens_effects(img, ca_px=2.0)
    assert np.array_equal(fringed[1], img[1])  # green stays put
    assert not np.array_equal(fringed[0], img[0]) and not np.array_equal(fringed[2], img[2])
    assert np.abs(fringed[:, 32, 48].astype(int) - img[:, 32, 48]).max() <= 2  # centre: little
    squeezed = lens_effects(img, jpeg_quality=40)
    assert squeezed.shape == img.shape and 0 < np.abs(squeezed.astype(int) - img).mean() < 25


def test_lens_effects_golden_output() -> None:
    import hashlib

    from fraktal.kernels.sensor import lens_effects

    out = lens_effects(_picture(), vignette=0.3, ca_px=1.2, jpeg_quality=70)
    assert hashlib.sha256(out.tobytes()).hexdigest()[:16] == "5692cd3ec8aa09cf"


def test_exposure_and_noise_are_neutral_at_zero_and_move_the_image() -> None:
    from fraktal.kernels.sensor import add_sensor_noise, expose

    img = _picture()
    assert np.array_equal(expose(img, 0.0, 0.0), img)
    assert expose(img, 1.0, 0.0).mean() > img.mean() > expose(img, -1.0, 0.0).mean()
    soft = expose(np.full((3, 4, 4), 250, dtype="uint8"), 1.0, 0.5)
    assert (
        soft.max() <= 255 and soft.min() > 200
    )  # the shoulder keeps highlights from clipping flat
    assert np.array_equal(add_sensor_noise(img, 0, 0, substream(0, "n")), img)
    noisy = add_sensor_noise(img, 2.0, 0.3, substream(0, "n"))
    assert np.array_equal(noisy, add_sensor_noise(img, 2.0, 0.3, substream(0, "n")))
    assert not np.array_equal(noisy, img)


def test_bonnet_covers_the_bottom_and_nothing_above_it() -> None:
    from fraktal.kernels.sensor import bonnet

    rgb, mask = bonnet(100, 160, 0.15, 0.1, (1.0, 1.0, 1.0))
    assert rgb.shape == (3, 100, 160) and mask.shape == (100, 160)
    assert not mask[:80].any() and mask[-1].all()
    assert 0.10 < mask.mean() < 0.17


def test_desaturate_goes_to_grey_and_is_neutral_at_one() -> None:
    from fraktal.kernels.sensor import desaturate

    img = _picture()
    assert np.array_equal(desaturate(img, 1.0), img)
    grey = desaturate(img, 0.0).astype("int16")
    assert np.abs(grey[0] - grey[1]).max() <= 1 and np.abs(grey[1] - grey[2]).max() <= 1
