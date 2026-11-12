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


def test_a_seeded_tile_may_draw_another_earth_tone_and_no_seed_keeps_the_base() -> None:
    base = soil_style("khartoum")
    assert soil_style("khartoum", None) is base
    tones = {soil_style("khartoum", seed).soil_a for seed in range(80)}
    assert base.soil_a in tones and len(tones) >= 3  # the base and at least two others
    for seed in range(20):
        drawn = soil_style("khartoum", seed)
        assert drawn.share == base.share and drawn.speckle == base.speckle  # only the colour moves
        assert drawn == soil_style("khartoum", seed)


def test_oil_palm_paddy_and_mangrove_each_have_their_own_ground() -> None:
    palm, paddy, coast = (
        soil_style(b) for b in ("oil_palm_plantation", "paddy_terraces", "mangrove_coast")
    )
    assert palm.bare_is_soil and paddy.bare_is_soil and coast.bare_is_soil
    assert len({palm.soil_a, paddy.soil_a, coast.soil_a}) == 3
    for seed in range(30):  # mud stays mud: no tone draw for the wet biomes
        assert soil_style("paddy_terraces", seed).soil_a == paddy.soil_a
        assert soil_style("mangrove_coast", seed).soil_a == coast.soil_a
    assert sum(soil_style("paddy_terraces", 1).soil_a) < sum(palm.soil_a)  # darker than the earth


def test_sealed_city_streets_take_asphalt_and_other_biomes_keep_their_road() -> None:
    import pytest

    from fraktal import ground_soil
    from fraktal.render import ROAD_ARTERIAL, ROAD_LOCAL

    surface = np.array([[ROAD_LOCAL, ROAD_ARTERIAL], [30, ROAD_LOCAL]], dtype="uint8")
    ground = np.full((2, 2, 3), 0.6, dtype="float32")
    sealed = ground_soil.seal_streets(ground, surface, "manila_quezon", ROAD_LOCAL)
    assert sealed[0, 0] == pytest.approx(np.array(ground_soil.SEALED_LOCAL_RGB) / 255.0)
    assert sealed[1, 1] == pytest.approx(sealed[0, 0])
    assert (sealed[0, 1] == 0.6).all() and (sealed[1, 0] == 0.6).all()  # arterial, grass: unchanged
    assert ground_soil.seal_streets(ground, surface, "khartoum", ROAD_LOCAL) is ground
    assert max(ground_soil.SEALED_LOCAL_RGB) / 255.0 < 0.4  # darker than the dusty street colour
