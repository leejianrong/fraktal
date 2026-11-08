import numpy as np

from fraktal.ground_soil import BIOME_SOIL, DEFAULT_SOIL, apply_soil, soil_style

SIDE = 200


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    surface = np.full((SIDE, SIDE), 30, dtype="uint8")  # grass
    surface[:, :40] = 10  # trees
    surface[:40, 100:] = 80  # water
    ground = np.full((SIDE, SIDE, 3), (0.5, 0.65, 0.3), dtype="float32")
    return ground, surface


def test_deterministic_and_seed_changes_it() -> None:
    ground, surface = _inputs()
    a = apply_soil(ground, surface, 3, 0.5, DEFAULT_SOIL)
    b = apply_soil(ground, surface, 3, 0.5, DEFAULT_SOIL)
    c = apply_soil(ground, surface, 4, 0.5, DEFAULT_SOIL)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_only_soil_capable_ground_changes() -> None:
    ground, surface = _inputs()
    out = apply_soil(ground, surface, 1, 0.5, DEFAULT_SOIL)
    untouched = ~np.isin(surface, (20, 30, 40))
    assert np.array_equal(out[untouched], ground[untouched])
    assert not np.array_equal(out[~untouched], ground[~untouched])


def test_share_of_soil_matches_the_style() -> None:
    ground, surface = _inputs()
    style = DEFAULT_SOIL
    out = apply_soil(ground, surface, 2, 0.5, style)
    capable = np.isin(surface, (20, 30, 40))
    red_over_green = out[..., 0] > out[..., 1]  # soil is redder than green; the grass is not
    assert abs(red_over_green[capable].mean() - style.share) < 0.08


def test_edges_are_crisp_not_blurred() -> None:
    ground, surface = _inputs()
    out = apply_soil(ground, surface, 5, 0.5, DEFAULT_SOIL)
    capable = np.isin(surface, (20, 30, 40))
    redness = out[..., 0] - out[..., 1]  # grass about -0.065, soil above 0.14; between is mixed
    mixed = (redness > -0.03) & (redness < 0.12) & capable
    assert mixed[capable].mean() < 0.12


def test_every_biome_has_a_style_and_unknown_falls_back() -> None:
    assert all(0 < s.share < 1 for s in BIOME_SOIL.values())
    assert soil_style("no-such-biome") is DEFAULT_SOIL


def test_khartoum_is_tan_and_mostly_soil() -> None:
    style = soil_style("khartoum")
    r, g, b = style.soil_a
    assert r > g > b and 0.5 < r < 0.75 and r - b > 0.2  # red-brown earth, not pale cream
    assert style.share >= 0.7
