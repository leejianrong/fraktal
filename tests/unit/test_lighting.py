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


def test_street_variety_has_its_own_stream_and_leaves_the_old_light_alone() -> None:
    d = lighting.draw(4, 50.0)
    assert d["variety"] == lighting.variety(4, 50.0)
    without = {k: v for k, v in d.items() if k != "variety"}
    assert without["sun_strength"] == lighting.draw(4, 50.0)["sun_strength"]
    assert d["variety"] != lighting.draw(5, 50.0)["variety"]


def test_street_variety_stays_in_its_stated_ranges() -> None:
    lo, hi = lighting.SUN_OFFSET_DEG
    for seed in range(60):
        v = lighting.variety(seed, 50.0)
        assert lo - 1e-3 <= v["sun_offset_deg"] <= hi + 1e-3
        assert lighting.CLOUD_COVER[0] <= v["cloud"]["cover"] <= lighting.CLOUD_COVER[1]
        assert lighting.CLOUD_SCALE_M[0] <= v["cloud"]["scale_m"] <= lighting.CLOUD_SCALE_M[1]
        assert lighting.HAZE_DENSITY[0] <= v["haze_density"] <= lighting.HAZE_DENSITY[1]
        assert all(0.0 <= c <= 1.0 for c in v["sun_colour"])


def test_a_lower_sun_is_warmer_and_khartoum_is_hazier() -> None:
    seeds = sorted(range(60), key=lambda s: lighting.variety(s, 50.0)["sun_offset_deg"])
    low, high = seeds[:15], seeds[-15:]
    assert sum(lighting.variety(s, 50.0)["sun_colour"][2] for s in low) < sum(
        lighting.variety(s, 50.0)["sun_colour"][2] for s in high
    )
    plain = [lighting.variety(s, 50.0)["haze_density"] for s in range(30)]
    hazy = [lighting.variety(s, 50.0, "khartoum_city")["haze_density"] for s in range(30)]
    assert min(hazy) >= lighting.HAZY_DENSITY["khartoum_city"][0]
    assert min(hazy) > max(plain)  # even the clearest Khartoum day is hazier than any other city's


def test_overhead_light_carries_a_lower_fill_and_a_stronger_sun() -> None:
    from fraktal import lighting

    sky = lighting.draw(7, 50.0)
    assert sky["overhead_fill"] < 1.0 < sky["overhead_sun"]
