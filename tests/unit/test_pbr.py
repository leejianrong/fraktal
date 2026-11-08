import numpy as np

from fraktal import pbr

SIZE = 128


def _ribs(period: int = 16) -> np.ndarray:
    """A tileable ribbed image (a sine across x), as an RGB uint8 array."""
    x = np.arange(SIZE)
    wave = 0.5 + 0.35 * np.sin(2 * np.pi * x / period)
    return np.repeat(np.repeat((wave[None, :] * 255).astype("uint8")[..., None], SIZE, 0), 3, 2)


def test_a_flat_image_gives_a_flat_normal_and_even_roughness() -> None:
    flat = np.full((SIZE, SIZE, 3), 120, dtype="uint8")
    normal, rough = pbr.derive(flat)
    assert (normal[..., 2] >= 254).all() and abs(int(normal[..., 0].mean()) - 128) <= 1
    assert rough.std() < 1.0


def test_ribs_tilt_the_normal_across_x_only_and_normals_are_unit_length() -> None:
    normal, _ = pbr.derive(_ribs())
    n = normal.astype("float64") / 255.0 * 2 - 1
    assert np.abs(np.linalg.norm(n, axis=-1) - 1).max() < 0.03
    assert n[..., 0].std() > 5 * n[..., 1].std()  # the slope runs across the ribs
    assert n[..., 0].min() < -0.2 and n[..., 0].max() > 0.2


def test_the_result_wraps_without_a_seam() -> None:
    normal, rough = pbr.derive(_ribs())
    # a sine's one-sample step can top its mean step by phase alone: compare with the input
    source = pbr.seam_score(_ribs())
    assert pbr.seam_score(rough) < 1.25 * max(source, 1.0)
    assert pbr.seam_score(normal[..., 0]) < 1.25 * max(source, 1.0)
    shifted = np.roll(_ribs(), 7, axis=1)  # a tileable map shifted is the same material
    n2, _ = pbr.derive(shifted)
    assert np.abs(np.roll(normal, 7, axis=1).astype(int) - n2.astype(int)).max() <= 1


def test_roughness_is_bounded_and_brighter_means_smoother() -> None:
    rgb = _ribs()
    _, rough = pbr.derive(rgb, rough_base=0.7)
    assert rough.min() >= 0.05 * 255 - 1 and rough.max() <= 255
    bright = pbr.lightness(rgb) > 0.8
    dark = pbr.lightness(rgb) < 0.3
    assert rough[bright].mean() < rough[dark].mean()


def test_seam_score_flags_a_hard_edge() -> None:
    ramp = np.tile(np.linspace(0, 1, SIZE), (SIZE, 1))  # does not wrap: a step at the seam
    assert pbr.seam_score(ramp) > 20
