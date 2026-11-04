import numpy as np
import pytest

from fraktal import heights


def test_the_storey_mix_follows_the_profile_and_keeps_the_rank_order() -> None:
    n = 2000
    raw = np.random.default_rng(0).uniform(3, 40, n)
    ids = [f"b{i:05d}" for i in range(n)]
    out = heights.remap(raw, ids, "lowrise", seed=1)
    storeys = np.floor(out / heights.STOREY_M).astype(int)
    for count, share in heights.PROFILES["lowrise"]:
        assert abs((storeys == count).mean() - share) < 0.01
    assert out.min() >= 3.0 and out.max() < 5 * 3.0 + 0.8 + 1e-9
    order = np.argsort(raw)
    assert (np.diff(storeys[order]) >= 0).all()  # a taller original is never fewer storeys


def test_remap_is_deterministic_and_does_not_depend_on_input_order() -> None:
    raw = np.array([5.0, 20.0, 9.0, 14.0, 7.0, 11.0])
    ids = ["a", "b", "c", "d", "e", "f"]
    one = heights.remap(raw, ids, "informal", seed=3)
    assert (one == heights.remap(raw, ids, "informal", seed=3)).all()
    perm = [3, 0, 5, 1, 4, 2]
    shuffled = heights.remap(raw[perm], [ids[i] for i in perm], "informal", seed=3)
    assert (shuffled == one[perm]).all()


def test_biomes_pick_a_profile_and_dense_urban_keeps_world_heights() -> None:
    assert heights.profile_for("khartoum") == "lowrise"
    assert heights.profile_for("informal_settlement") == "informal"
    assert heights.profile_for("dense_urban") is None


def test_bad_input_is_refused() -> None:
    with pytest.raises(heights.HeightError):
        heights.remap([1.0], ["a"], "skyscraper", 0)
    with pytest.raises(heights.HeightError):
        heights.remap([1.0, 2.0], ["a"], "lowrise", 0)
