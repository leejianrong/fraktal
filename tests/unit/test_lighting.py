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
