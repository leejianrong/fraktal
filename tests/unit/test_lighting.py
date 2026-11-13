from fraktal import lighting


def test_light_is_deterministic_and_varies_by_seed() -> None:
    a = lighting.draw(1, 50.0)
    assert a == lighting.draw(1, 50.0) and a != lighting.draw(2, 50.0)
    assert len({tuple(lighting.draw(s, 50.0)["horizon"]) for s in range(20)}) > 10  # dust varies


def test_a_low_sun_is_warmer_and_fill_stays_in_range() -> None:
    low = [lighting.draw(s, 15.0)["sun_colour"] for s in range(30)]
    high = [lighting.draw(s, 65.0)["sun_colour"] for s in range(30)]

    def mean_blue(cs: list[list[float]]) -> float:
        return sum(c[2] for c in cs) / len(cs)

    assert mean_blue(low) < mean_blue(high) - 0.15
    assert all(c[0] >= c[1] >= c[2] for c in low + high)  # always warm, never blue
    for s in range(30):
        d = lighting.draw(s, 40.0)
        assert lighting.FILL_RANGE[0] <= d["fill"] <= lighting.FILL_RANGE[1]
        assert all(0.0 <= v <= 1.0 for key in ("zenith", "horizon", "glow") for v in d[key])


def test_the_light_carries_the_eye_level_factors_that_darken_the_street_view() -> None:
    eye = lighting.draw(3, 50.0)["eye_level"]
    assert eye == lighting.EYE_LEVEL and eye["sun"] < 1.0 and eye["fill"] < 1.0
    assert eye["sky_seen"] < lighting.FILL_RANGE[0]  # the visible sky is dimmer than its light


def test_a_hazy_city_has_a_paler_sky_and_nairobi_keeps_more_light() -> None:
    plain, hazy = lighting.draw(5, 50.0), lighting.draw(5, 50.0, "khartoum_city")
    assert hazy["horizon"][2] < plain["horizon"][2]  # less blue: dust in the air
    assert hazy["zenith"][2] < plain["zenith"][2]
    nairobi = lighting.draw(5, 50.0, "nairobi_centre")["eye_level"]
    assert (
        nairobi["sun"] > lighting.EYE_LEVEL["sun"] and nairobi["fill"] > lighting.EYE_LEVEL["fill"]
    )
    assert lighting.draw(5, 50.0, "dense_urban") == plain  # other biomes are unchanged
